{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE BangPatterns #-}

module Main
    ( main
    ) where
        
import           Codec.Binary.UTF8.String (encode)
import           Data.Aeson               (FromJSON (..), ToJSON (..),
                                           eitherDecode)
import qualified Data.ByteString.Internal as B (c2w)
import qualified Data.ByteString.Lazy     as B
import           GHC.Generics
import           System.Random.Shuffle    (shuffle')
import           System.Random            (newStdGen)
import           Torch.Autograd           (makeIndependent, toDependent)

import           Torch.NN                 (Parameter, Parameterized (..),
                                           Randomizable (..))
import           Torch.Serialize          (loadParams)
import           Torch.TensorFactories    (randnIO', zeros', ones')
import           Torch.Device
import           Torch.Functional         (mseLoss, sumAll, stack, Dim(..))
import           Torch.Tensor             (Tensor, asTensor, asValue)
import           Torch.Train              (saveParams)

import           ML.Exp.Chart                 (drawLearningCurve)
import           Word2Vec                 (EmbeddingParams (..), EmbeddingHypParams (..)) 

import           Torch.Layer.NonLinear    (ActName (..))
import           Torch.Optim                   (Adam (..), GD (..), foldLoop,
                                               mkAdam, runStep)
import           Torch.Layer.RNN           (RnnHypParams(..), RnnParams(..), rnnLayers)
import           Torch.Tensor.Util         (unstack)

import           Torch.Layer.Linear        (LinearHypParams(..), LinearParams(..), linearLayer)
import Text.CSV
import TextTreatement (parseCsvQuoted)

import Control.Monad        (when)
import Debug.Trace  (trace)

data CurrencyData = CurrencyData
    { date :: String,
      price :: Float,
      open :: Float,
      high :: Float,
      low :: Float,
      vol :: Float,
      change :: Float
    }
    deriving (Show, Generic)

data ModelSpec =
    ModelSpec
        { inputDim :: Int,
          hiddenDim :: Int,
          outputDim :: Int,
          numLayers :: Int
        }
    deriving (Show, Eq, Generic)

data Model = 
    Model
        { rnn :: RnnParams,
          outputLayers :: LinearParams
        }
    deriving (Show, Generic, Parameterized)

instance Randomizable ModelSpec Model where
    sample ModelSpec {..} = do
        rnn_sampled <- sample $ RnnHypParams {dev = Device CPU 0, bidirectional = False, inputSize = inputDim, hiddenSize = hiddenDim, numLayers = numLayers, hasBias = True}
        output_sampled <- sample $ LinearHypParams {dev = Device CPU 0, inputDim = hiddenDim, outputDim = outputDim, hasBias = True}
        return Model {rnn = rnn_sampled, outputLayers = output_sampled}
            
forward :: Model -> Tensor -> Tensor
forward model input = stack (Dim 0) $ map (linearLayer (outputLayers model)) (unstack rnnOutputs) 
    where rnnOutputs = fst $ rnnLayers (rnn model) Relu (Just 0.8) (ones' [4, 64]) input


removeUglyChar :: String -> Char -> String
removeUglyChar [] char = []
removeUglyChar (c:cs) char = if c == char then removeUglyChar cs char else c : removeUglyChar cs char

createTrainingData :: [CurrencyData] -> Int -> [(Tensor, Tensor)]
createTrainingData [] _ = []
createTrainingData [c] _ = []
createTrainingData (currency:currencyDatas) daysNb = if length currencyDatas >= (daysNb + 1) then trainingData : (createTrainingData currencyDatas daysNb) else []
    where trainingData = (asTensor input, asTensor output)
          output = tail datas
          input = init datas
          datas = map (\prevCurrency -> [price prevCurrency]) (take (daysNb+1) currencyDatas)
        --   datas = map (\prevCurrency -> [price prevCurrency, open prevCurrency, high prevCurrency, low prevCurrency]) (take (daysNb+1) currencyDatas)

extractData :: [[String]] -> [CurrencyData]
extractData csvData = [extract line | line <- csvData, line !! 5 /= "-"]
    where 
          extract line = CurrencyData {date = line !! 0, price = read (removeUglyChar (line !! 1 ) ',') :: Float , open = read (removeUglyChar (line !! 2) ',') :: Float, high = read (removeUglyChar (line !! 3) ',') :: Float, low = read (removeUglyChar (line !! 4) ',') :: Float, vol = volume line , change = read (removeUglyChar (init $ line !! 6) ',') :: Float}
          volume line= read (removeUglyChar (init $ line !! 5) ',') :: Float

filterCSV :: [[String]] -> [CurrencyData]
filterCSV [] = []
filterCSV (_:csvData) = extractData csvData

inputTrainingDataLength :: Int
inputTrainingDataLength = 1

daysNb :: Int
daysNb = 40

-- forward :: Model -> Tensor -> Tensor
-- forward model input = fst $ rnnLayers model Relu (Just 0.8) (ones' [4, inputTrainingDataLength]) input

loss :: Model -> (Tensor, Tensor) -> Tensor
loss model (input, target) = (sumAll $ mseLoss (forward model input) target)


-- accuracy : check if the model predicted the curve to go up or down 
accuracy :: Model -> (Model -> Tensor -> Tensor) -> [(Tensor, Tensor)] -> Float
accuracy model forward trainingDatas = res / fromIntegral (length trainingDatas)
    where res = sum $ map (\(input, target) -> check input target) trainingDatas
          check input target = if (guessCurveDirection input) * (curveDirection input) >= (0 :: Float) then 1 else 0
          guessPrice input = (head $ last $ asValue $ forward model input) 
          guessCurveDirection input = (guessPrice input) - (head $ last $ init $ asValue $ forward model input)
          curveDirection target =  (head $ last $ init $ asValue target) - (head $ last $ asValue target)

main :: IO ()
main = do
    !csvLines <- parseCsvQuoted "/data/trading/bitcoin_history.csv"
    putStrLn $ "CSV lines grabbed : " ++ show (length csvLines)
    let currencyDatas = reverse $ filterCSV csvLines
        trainingDatas = createTrainingData (drop (length currencyDatas - 2000) currencyDatas) daysNb
        hypParams = ModelSpec {inputDim = inputTrainingDataLength, hiddenDim = 64, numLayers = 4, outputDim = inputTrainingDataLength}
    initModel <- sample $ hypParams

    model <- trainModel initModel trainingDatas 50 currencyDatas
    model <- loadParams initModel "app/trading/models/trading_40_83%_1282422734848loss.model"

    -- print $ (stack (Dim 0) $ take 100 $ map currencyToTensor currencyDatas)
    let nextPrices = guessNextPrices model (asTensor $ take 1000 $ map (\x -> [price x]) currencyDatas) 1000
        currencyPrices = map price currencyDatas
        guesses = map price nextPrices
    print guesses
    drawLearningCurve "app/trading/models/graph-currency2.png" "Currency Price" [("guesses", currencyPrices++ guesses),("actual", currencyPrices )]

    -- print $ forward model (asTensor ([[0.0], [0.0], [0.0], [0.0]] :: [[Float]]))
    return ()

trainModel :: Model -> [(Tensor, Tensor)] -> Int -> [CurrencyData] -> IO Model
trainModel initmodel trainingDatas epochNb currencyDatas = do
    putStrLn "Training model..."
    rng <- newStdGen
    let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initmodel)
        validationSetLength = length trainingDatas `div` 10
        (validation, training) = splitAt validationSetLength (shuffle' trainingDatas (length trainingDatas) rng)
    -- let optimizer = GD
    (trainedModel, _, losses) <-
        foldLoop (initmodel, optimizer, []) epochNb $ \(model, opt, losses) i -> do
            let epochLoss = sum (map (loss model) training)
            let lossValue = asValue epochLoss :: Float
            putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue
            (trainedModel, nOpt) <- runStep model opt epochLoss 0.001
            when (i `mod` 10 == 0) $ do
                putStrLn "Saving..."
                saveParams trainedModel ("app/trading/models/trading_" ++ show (i + 0) ++ "_" ++ show (round (100 * (accuracy model forward validation))) ++ "%_" ++ show (round lossValue) ++ "loss.model" )
                drawLearningCurve "app/trading/models/graph-trading.png" "Learning Curve" [("learningRate", losses)]
                putStrLn "Saved..."
                -- let currencyPrices = reverse $ map price currencyDatas
                --     guesses = reverse $ map (\(input, _) -> asValue $ forward model input) training
                -- drawLearningCurve "app/trading/models/graph-currency.png" "Currency Price" [("price", currencyPrices),("price", (replicate (4956-(length trainingDatas)) 0) ++ guesses)]
                

            pure (trainedModel, nOpt, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it
    -- saveParams trainedModel modelPath
    return trainedModel

guessNextPrices :: Model -> Tensor -> Int -> [CurrencyData] 
guessNextPrices model input nb = guessNextPrices' model input nb []
    where guessNextPrices' model input' 0 res = res  
          guessNextPrices' model input' nb res = guessNextPrices' model output (nb - 1) (res ++ [tensorToCurrency output])
              where output = forward model input'
          tensorToCurrency x = toCurrency $ last $ (asValue x :: [Float])
          toCurrency price = CurrencyData {date = "", price = price, open = 0, high = 0, low = 0, vol = 0, change = 0}
        
currencyToTensor :: CurrencyData -> Tensor
currencyToTensor = asTensor . price