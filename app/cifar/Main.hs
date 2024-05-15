-- PLEASE DOWNLOAD THE REQUIRED DATAS IF YOU WANT TO RUN THIS PROGRAM 
-- https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download

module Main where

import Functional           (sortBySnd)
import Torch.Model.Utils    (accuracy, f1, precision, recall, macroAvg, weightedAvg)
import Torch.Tensor.Util    (indexOfMax, oneHot')
import qualified Data.Text as T

import Graphics.Matplotlib
import System.Random
import Data.List.Split

import Torch.Optim          (foldLoop)
import ReadImage            (imageToRGBList)

import Control.Monad        (when)
import Data.List            (sortBy, maximumBy)

import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.Functional     (mseLoss, Dim(..), exp, sumAll, div)
import Torch.NN             (sample,flattenParameters)
import Torch.Optim          (GD(..), Adam(..), mkAdam, runStep, foldLoop)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP      (MLPHypParams(..), MLPParams(..), ActName(..), mlpLayer)
import ML.Exp.Chart         (drawLearningCurve,drawConfusionMatrix ) --nlp-tools

import Text.CSV
import Data.Csv
import qualified Data.ByteString.Lazy as BL

getObjectData :: String -> Int -> Int -> IO [(Tensor, Tensor)]
getObjectData folderName output maxNb = do
    datas <- foldLoop [] 1000 $ \currentDatas i -> do 
        let fileName = show i
            paddingZero = concat $ take (4 - length fileName) ["0", "0", "0", "0"]
            imagePath = "data/cifar10/" ++ folderName ++ "/" ++ paddingZero ++ fileName ++ ".png"
        -- print imagePath
        image <- imageToRGBList imagePath
        case image of
            Left _ -> pure currentDatas
            Right rgbData -> do 
                let outputData = (take output (repeat 0 :: [Float])) ++ [1.0] ++ (take (maxNb - output - 1) (repeat 0 :: [Float]))
                let tensorOutput = asTensor outputData
                pure (currentDatas ++ [(asTensor rgbData, tensorOutput)])
    return datas

getData :: String -> IO [(Tensor, Tensor)]
getData folderName = do
    let outputSize = 10
    airplane <- getObjectData (folderName ++ "airplane") 0 outputSize
    automobile <- getObjectData (folderName ++ "automobile") 1 outputSize
    bird <- getObjectData (folderName ++ "bird") 2 outputSize
    cat <- getObjectData (folderName ++ "cat") 3 outputSize
    deer <- getObjectData (folderName ++ "deer") 4 outputSize
    dog <- getObjectData (folderName ++ "dog") 5 outputSize
    frog <- getObjectData (folderName ++ "frog") 6 outputSize
    horse <- getObjectData (folderName ++ "horse") 7 outputSize
    ship <- getObjectData (folderName ++ "ship") 8 outputSize
    truck <- getObjectData (folderName ++ "truck") 9 outputSize
    return $ concat $ [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
    
loss :: MLPParams -> (Tensor, Tensor) -> Tensor
loss model (input, output) = let y = forward model input
                             in mseLoss y output

forward :: MLPParams -> Tensor -> Tensor
forward model input = mlpLayer model input 

main :: IO ()
main = do
    let device = Device CPU 0
        epochNb = 10000 
        hypParams = MLPHypParams device 3072 [(256, Relu),(256, Relu),(10, Softmax)] -- Id | Sigmoid | Tanh | Relu | Elu | Selu

    -- putStrLn "grabing data..."
    -- trainingData <- getData "train/"
    validationData <- getData "test/"
    putStrLn "data grabbed"
    model <- sample hypParams
    print validationData
    -- initModel <- loadParams hypParams "app/cifar/models/trainingCifar_256_256/cifar_700_51%_2881loss.model"
    -- putStrLn "start training"
    -- let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initModel)
    -- (trainedModel, _, losses) <- foldLoop (initModel, optimizer, []) epochNb $ \(model, opt, losses) i -> do 
    --     let epochLoss = sum (map (loss model) trainingData)
    --     let lossValue = asValue epochLoss :: Float
    --     putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue 
    --     (trainedModel, nOpt) <- runStep model opt epochLoss 0.002
    --     when (i `mod` 25 == 0) $ do
    --         putStrLn "Saving..."
    --         saveParams trainedModel ("app/cifar/models/trainingCifar_256_256/cifar_" ++ show (i + 700) ++ "_" ++ show (round (100 * (accuracy model forward trainingData))) ++ "%_" ++ show (round lossValue) ++ "loss.model" )
    --         drawLearningCurve "app/cifar/models/graph-cifar.png" "Learning Curve" [("", losses)]
    --         putStrLn "Saved..."
    --     pure (trainedModel, nOpt, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it 
    -- drawLearningCurve "app/cifar/models/graph-cifar.png" "Learning Curve" [("", losses)]
    -- saveParams trainedModel "app/cifar/models/cifar.model"
    
    -- model <- loadParams hypParams "app/cifar/models/trainingCifar_256_256/cifar_2300_89%_181loss.model"
    putStrLn "model loaded"    
    -- show confusion matrix with placeholder data
    let results = map (\(input, output) -> (lookTable !! (indexOfMax $ (asValue (oneHot' $ forward model input) :: [Float])), lookTable !! (indexOfMax $ (asValue output :: [Float])))) validationData
    print $ recall model forward validationData
    print $ precision model forward validationData
    print $ f1 model forward validationData
    print $ macroAvg model forward validationData
    print $ weightedAvg model forward validationData
    drawConfusionMatrix "app/cifar/confusion_matrix4.png" 10 results
    

    -- let confMat = confusionMatrix model forward validationData
    -- putStrLn "confmat calculated"
    -- file "app/cifar/confusion_matrix3.png" $ confusionMatrixPlot confMat lookTable


    return ()


getObjectDataTesting :: String -> Int -> IO [Tensor]
getObjectDataTesting folderName itt = do
    datas <- foldLoop [] 10000 $ \currentDatas i -> do 
        let fileName = show (i + itt)
            imagePath = "data/test/" ++ fileName ++ ".png"
        image <- imageToRGBList imagePath
        case image of
            Left _ -> pure currentDatas
            Right rgbData -> do 
                pure (currentDatas ++ [asTensor rgbData])
    return datas

getDataTesting :: Int -> IO [Tensor]
getDataTesting itt = do
    res <- getObjectDataTesting "test/" itt
    return res

main_s :: IO ()
main_s = do
    let device = Device CPU 0
        epochNb = 10000 
        hypParams = MLPHypParams device 3072 [(256, Relu),(256, Relu),(10, Softmax)] -- Id | Sigmoid | Tanh | Relu | Elu | Selu

    model <- loadParams hypParams "app/cifar/models/trainingCifar_256_256/cifar_1000_65%_495loss.model"

    datas <- foldLoop [] 30 $ \currentDatas i -> do 
        putStrLn $ show i 
        validationData <- getDataTesting ((i-1)*10000)
        let res = map (forward model) validationData
        let calculated = [(asValue value :: [Float]) | value <- res]
        let sorted = map (\x -> reverse $ sortBySnd $ zip lookTable x) calculated
        let names = map (\x -> fst (x !! 0)) sorted
        let output = zip [(1+((i-1)*10000))..] names
        let csvOutput = map (\(id, prediction) -> [show id, prediction]) output
        BL.writeFile ("outputs/cifar_" ++ show i ++ ".csv")  $ encode csvOutput
        return []

    -- image <- imageToRGBList "data/test/300000.png" 
    -- case image of
    --     Left _ -> return []
    --     Right rgbData -> do 
    --         let inputData = asTensor rgbData
    --         let calculated = forward model inputData
    --         print calculated
    --         let guess = asValue calculated :: [Float]
    --         let both = zip lookTable guess
    --         let sorted = reverse $ sortBySnd both
    --         print sorted
    --         return []
    return ()

lookTable :: [String]
lookTable = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]