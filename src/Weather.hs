module Weather (weather) where

import Text.CSV
import System.Random (newStdGen)
import System.Random.Shuffle (shuffle')
import System.Directory (getCurrentDirectory)


import Torch.Functional     (mseLoss)
import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.NN             (sample)
import Torch.Optim          (GD(..), runStep, foldLoop)

import Torch.Train          (update, saveParams, loadParams)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Layer.Linear   (LinearHypParams(..), LinearParams(..), linearLayer)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools


-- Function to parse CSV file or throw an error
parseCSVOrError :: String -> IO CSV
parseCSVOrError filePath = do
    currentDir <- getCurrentDirectory
    csv <- parseCSVFromFile (currentDir ++ "/" ++ filePath)
    case csv of
        Right csvData -> return csvData
        Left err -> error $ "Error parsing CSV: " ++ show err


createModel :: Device -> LinearHypParams
createModel device = LinearHypParams device True 11 1

forward :: LinearParams -> [Float] -> Tensor
forward model input = linearLayer model $ asTensor input'
    where input' = zipWith (\x i -> x ^ i) input [1..] -- non linearity

loss :: LinearParams -> ([Float], Float) -> Tensor
loss model (input, output) = let y = forward model input 
                             in mseLoss y (asTensor output)

slopeTrend :: [Float] -> Float
slopeTrend xs = sumOfDifferences / fromIntegral (length xs - 1)
    where differences = zipWith (-) (tail xs) xs 
          sumOfDifferences = sum differences 

extractTemperatures :: CSV -> [Float]
extractTemperatures (header:csv) = [ (read (line !! 1) :: Float)/30.0| line <- csv]

retreivePackOfSevenTemperatures :: [Float] -> [([Float],Float)]
retreivePackOfSevenTemperatures temperatures = if (length temperatures <= 10) 
                                               then [(fullData, output)]
                                               else [(fullData, output)] ++ retreivePackOfSevenTemperatures (tail temperatures)
                                               where output = temperatures !! 8
                                                     week = take 7 temperatures
                                                     mean = (sum week) / 7.0
                                                     variance = 1.0/6.0 * sum (map (\x -> (x - mean) ** 2) week) -- 1/(n-1)
                                                     weekend = drop ((length week) - 2 ) week
                                                     diff2lastDays = (weekend !! 0) - (weekend !! 1)
                                                     fullData = week ++ [variance, diff2lastDays, slopeTrend week, mean]

getNextBatch :: Int -> [([Float], Float)] -> [([Float], Float)] 
getNextBatch n l = drop n l ++ take n l

device = Device CPU 0

train :: [([Float], Float)] -> IO()
train trainingDatas = do 
    -- sample initiate a network with random value with the size of our model
    initialModel <- sample $ createModel device

    -- the foldLoop is the epoch loop
    (trainedModel, _, _, _, losses) <- foldLoop (initialModel, optimizer, trainingDatas, batchSize, []) epochNb $ \(model, opt, datas, batchSize, losses) i -> do 
        let epochLoss = sum (map (loss model) (take batchSize datas))
        let lossValue = asValue epochLoss :: Float
        putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue
        (trainedModel, nOpt) <- runStep model opt epochLoss 1e-4
        pure (trainedModel, nOpt, getNextBatch 100 datas, batchSize, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it 
    -- print trainedModel
    drawLearningCurve "models/graph-weather.png" "Learning Curve" [("", losses)]

    saveParams trainedModel "models/weather.model"

    where optimizer = GD
          epochNb = 200
          batchSize = 1000

weather :: IO()
weather = do
    rng <- newStdGen
    currentDir <- getCurrentDirectory
    csvTrain <- parseCSVOrError "/data/weather/train.csv"
    let temperatures = extractTemperatures csvTrain
    let parsedDatas = retreivePackOfSevenTemperatures temperatures
    let trainingDatas = shuffle' parsedDatas (length parsedDatas) rng

    train trainingDatas

    csvValidation <- parseCSVOrError "/data/weather/valid.csv"
    let vTemperatures = extractTemperatures csvValidation
    let validationDatas = retreivePackOfSevenTemperatures vTemperatures

    model <- loadParams (createModel device) "models/weather.model"

    let validationOutputs = map (\(input, _) -> asValue (forward model input) :: Float) validationDatas
    
    -- zip expected and actual values to display
    putStrLn $ "(Actual, Expected) : " ++ show (asTensor $ map tupleToList2 $ zip validationOutputs (map snd validationDatas))
    let mse =  asValue (sum $ map (loss model) validationDatas) :: Float
    putStrLn $ "MSE : " ++ show  mse ++ " | MSE/dataNb : " ++ show (mse / (fromIntegral (length validationDatas)))
    --validation


-- convert tuple to list (not needed for the neural network, only for display purposes)
tupleToList2 :: (Float, Float) -> [Float]
tupleToList2 (x, y) = [x*30.0, y*30.0]