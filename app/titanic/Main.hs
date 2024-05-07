module Main where

import Text.CSV
import Data.Csv
import qualified Data.ByteString.Lazy as BL

import Utils
import Control.Monad (forM_)

-- import Debug.Trace
import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.Functional     (mseLoss)
import Torch.NN             (sample)
import Torch.Optim          (GD(..), foldLoop)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP (MLPHypParams(..), MLPParams(..), ActName(..), mlpLayer)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools



loss :: MLPParams -> (Tensor, Float) -> Tensor
loss model (input, output) = let y = mlpLayer model input 
                             in mseLoss y (asTensor output)


extractData :: [[String]] -> [(Tensor, Float)]
extractData csvData = [extract line | line <- csvData, length line == 12]
    where 
        extract line = (asTensor [pclass, sex, age, sibSp, parch, fare, embarqued], read (line !! 1) :: Float)
            where
                pclass = readFiltered (line !! 2)/3.0
                sex = readMaleFemale (line !! 4)
                age = readFiltered (line !! 5) / 10.0
                sibSp = readFiltered (line !! 6)
                parch = readFiltered (line !! 7)
                fare = readFiltered (line !! 9) / 100.0
                embarqued = readEmbarqued (line !! 11)
                readFiltered value = if value /= "" then (read value :: Float) else 0.0
                readMaleFemale value = if value == "male" then -1.0 else 1.0
                readEmbarqued value = if value == "S" then 1.0 else if value == "C" then 0.5 else 0.0

filterCSV :: CSV -> [(Tensor, Float)]
filterCSV [] = []
filterCSV (_:csvData) = extractData csvData

filterCSVValid :: CSV -> [Tensor]
filterCSVValid [] = []
filterCSVValid (_:csvData) = map fst $ extractData [take 1 line ++ ["0.0"] ++ drop 1 line  | line <- csvData]


main :: IO ()
main = do
    let device = Device CPU 0
        epochNb = 1000
        hypParams = MLPHypParams device 7 [(30, Relu),(4, Relu), (1, Id)] -- Id | Sigmoid | Tanh | Relu | Elu | Selu
    csvTrain <- parseCSVOrError "/data/titanic/train.csv"
    let trainingData = filterCSV csvTrain
    -- -- forM_ trainingData $ \(input,_) -> do
    -- --     putStrLn $ show $ input
    initModel <- sample hypParams
    -- initModel <- loadParams hypParams "models/titanic_7input_83-38%.model"
    (trainedModel, _, losses) <- foldLoop (initModel, GD, []) epochNb $ \(model, opt, losses) i -> do 
        let epochLoss = sum (map (loss model) trainingData)
        let lossValue = asValue epochLoss :: Float
        putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue 
        (trainedModel, nOpt) <- update model opt epochLoss 0.00001
        pure (trainedModel, nOpt, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it 
    drawLearningCurve "models/graph-titanic.png" "Learning Curve" [("", losses)]
    saveParams trainedModel "models/titanic.model"


    model <- loadParams hypParams "models/titanic.model"
    -- -- -- forM_ trainingData $ \(input,output) -> do
    -- -- --     putStr $ show $ output
    -- -- --     putStr ": "
    -- -- --     putStrLn $ show ((mlpLayer model input))

    let expected = map snd trainingData
    let calculated = [ if (asValue value :: Float) <= 0.5 then 0 else 1 | value <- map (mlpLayer model . fst) trainingData]
    let differences = map (\(x, y) -> 1-(abs (x - y))) (zip expected calculated)
    putStrLn $ "Res : " ++ show ((sum differences) / fromIntegral (length differences)) ++ "%"

    -- csvValid <- parseCSVOrError "/data/titanic/test.csv"
    -- let validationData = filterCSVValid csvValid
    -- let calculated = [ if (asValue value :: Float) <= 0.5 then 0 else 1 | value <- map (mlpLayer model) validationData]
    -- let passengersId = [read (line !! 0) :: Int | line <- drop 1 csvValid, length line == 11]
    -- let output = zip passengersId calculated

    -- let csvOutput = map (\(passengerId, prediction) -> [show passengerId, show prediction]) output
    -- BL.writeFile "output3.csv" $ encode csvOutput



    -- forM_ validationData $ \input -> do
    --     putStr $ show $ input
    --     putStr ": "
    --     putStrLn $ show ((mlpLayer trainedModel input))
    return ()
    
    