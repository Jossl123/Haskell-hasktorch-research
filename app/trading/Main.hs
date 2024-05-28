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
import           Torch.Autograd           (makeIndependent, toDependent)
import           Torch.NN                 (Parameter, Parameterized (..),
                                           Randomizable (..))
import           Torch.Serialize          (loadParams)
import           Torch.TensorFactories    (randnIO', zeros')
import           Torch.Device
import           Torch.Functional         (mseLoss, sumAll, div)
import           Torch.Tensor             (Tensor, asTensor, asValue)
import           Torch.Train              (saveParams)

import           Word2Vec                 (EmbeddingParams (..), EmbeddingHypParams (..)) 

import           Torch.Layer.NonLinear    (ActName (..))
import           Torch.Optim                   (Adam (..), GD (..), foldLoop,
                                               mkAdam, runStep)
import           Torch.Layer.RNN           (RnnHypParams(..), RnnParams(..), rnnLayers)

import Text.CSV
import TextTreatement (parseCsvQuoted)

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



removeUglyChar :: String -> Char -> String
removeUglyChar [] char = []
removeUglyChar (c:cs) char = if c == char then removeUglyChar cs char else c : removeUglyChar cs char

createTrainingData :: [CurrencyData] -> Int -> [(Tensor, Tensor)]
createTrainingData [] _ = []
createTrainingData [c] _ = []
createTrainingData (currency:currencyDatas) daysNb = if length currencyDatas >= (daysNb + 1) then trainingData : (createTrainingData currencyDatas daysNb) else []
    where trainingData = (asTensor input, asTensor output)
          output = [price currency]
          input = map (\prevCurrency -> [price prevCurrency]) (take daysNb currencyDatas)

extractData :: [[String]] -> [CurrencyData]
extractData csvData = [extract line | line <- csvData, line !! 5 /= "-"]
    where 
          extract line = CurrencyData {date = line !! 0, price = read (removeUglyChar (line !! 1 ) ',') :: Float , open = read (removeUglyChar (line !! 2) ',') :: Float, high = read (removeUglyChar (line !! 3) ',') :: Float, low = read (removeUglyChar (line !! 4) ',') :: Float, vol = volume line , change = read (removeUglyChar (init $ line !! 6) ',') :: Float}
          volume line= read (removeUglyChar (init $ line !! 5) ',') :: Float

filterCSV :: [[String]] -> [CurrencyData]
filterCSV [] = []
filterCSV (_:csvData) = extractData csvData

currentHiddenSize :: Int
currentHiddenSize = 4

loss :: RnnParams -> (Tensor, Tensor) -> Tensor
loss model (input, target) = (sumAll $ mseLoss output target) `Torch.Functional.div` 100000
    where
        (output, _) = rnnLayers model Tanh (Just 0.8) input (zeros' [1, currentHiddenSize])

main :: IO ()
main = do
    csvLines <- parseCsvQuoted "/data/trading/bitcoin_history.csv"
    let currencyDatas = filterCSV csvLines
        daysNb = 50
        trainingDatas = createTrainingData (take 100 currencyDatas) daysNb
    initModel <- sample $ RnnHypParams {dev = Device CPU 0, bidirectional = False, inputSize = 4, hiddenSize = currentHiddenSize, numLayers = daysNb, hasBias = True}
    model <- trainModel initModel trainingDatas 1000
    return ()

trainModel :: RnnParams -> [(Tensor, Tensor)] -> Int -> IO RnnParams
trainModel initmodel trainingDatas epochNb = do
    putStrLn "Training model..."
    let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initmodel)
    -- let optimizer = GD
    (trainedModel, _, losses) <-
        foldLoop (initmodel, optimizer, []) epochNb $ \(model, opt, losses) i -> do
            let epochLoss = sum (map (loss model) trainingDatas)
            let lossValue = asValue epochLoss :: Float
            putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue
            (trainedModel, nOpt) <- runStep model opt epochLoss 0.001
            pure (trainedModel, nOpt, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it
    -- saveParams trainedModel modelPath
    return trainedModel