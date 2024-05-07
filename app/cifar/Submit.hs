-- PLEASE DOWNLOAD THE REQUIRED DATAS IF YOU WANT TO RUN THIS PROGRAM 
-- https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download

module Submit where

import Text.CSV
import Data.Csv
import qualified Data.ByteString.Lazy as BL

import Torch.Optim          (foldLoop)
import ReadImage (imageToRGBList)

import Control.Monad (when)
import Data.List (sortBy, maximumBy)

import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.Functional     (mseLoss, Dim(..), exp, sumAll, div)
import Torch.NN             (sample,flattenParameters)
import Torch.Optim          (GD(..), Adam(..), mkAdam, runStep, foldLoop)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP (MLPHypParams(..), MLPParams(..), ActName(..), mlpLayer)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools

getObjectData :: String -> Int -> IO [Tensor]
getObjectData folderName itt = do
    datas <- foldLoop [] 10000 $ \currentDatas i -> do 
        let fileName = show (i + itt)
            imagePath = "data/test/" ++ fileName ++ ".png"
        image <- imageToRGBList imagePath
        case image of
            Left _ -> pure currentDatas
            Right rgbData -> do 
                pure (currentDatas ++ [asTensor rgbData])
    return datas

getData :: Int -> IO [Tensor]
getData itt = do
    res <- getObjectData "test/" itt
    return res
    

loss :: MLPParams -> (Tensor, Tensor) -> Tensor
loss model (input, output) = let y = forward model input
                             in mseLoss y output

forward :: MLPParams -> Tensor -> Tensor
forward model input = softmax $ mlpLayer model input 

softmax :: Tensor -> Tensor
softmax input = 
    let expInput =  Torch.Functional.exp input
        expSum = sumAll expInput
    in Torch.Functional.div expInput expSum

cifar :: IO ()
cifar = do
    let device = Device CPU 0
        epochNb = 10000 
        hypParams = MLPHypParams device 3072 [(256, Relu),(256, Relu),(10, Id)] -- Id | Sigmoid | Tanh | Relu | Elu | Selu

    -- trainingData <- getData "train/"
    -- initModel <- sample hypParams
    -- initModel <- loadParams hypParams "models/trainingCifar/cifar_200_17%_895loss.model"
    -- let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initModel)
    
    -- (trainedModel, _, losses) <- foldLoop (initModel, optimizer, []) epochNb $ \(model, opt, losses) i -> do 
    --     let epochLoss = sum (map (loss model) trainingData)
    --     let lossValue = asValue epochLoss :: Float
    --     putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue 
    --     (trainedModel, nOpt) <- runStep model opt epochLoss 0.001
    --     when (i `mod` 50 == 0) $ do
    --         putStrLn "Saving..."
    --         let results = map (\(input, output) -> if (indexOfMax $ (asValue (forward model input) :: [Float])) == (indexOfMax $ (asValue output :: [Float])) then 1 else 0) validationData
    --         let grade = ((sum results) / (fromIntegral (length results))) * 100.0
    --         saveParams trainedModel ("models/trainingCifar/cifar_" ++ show (i + 0) ++ "_" ++ show (round grade) ++ "%_" ++ show (round lossValue) ++ "loss.model" )
    --         drawLearningCurve "models/graph-cifar.png" "Learning Curve" [("", losses)]
    --         putStrLn "Saved..."
    --     pure (trainedModel, nOpt, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it 
    -- drawLearningCurve "models/graph-cifar.png" "Learning Curve" [("", losses)]
    -- saveParams trainedModel "models/cifar.model"



    model <- loadParams hypParams "models/trainingCifar/cifar_700_51%_2881loss.model"
    
    -- let results = map (\(input, output) -> if (indexOfMax $ (asValue (forward model input) :: [Float])) == (indexOfMax $ (asValue output :: [Float])) then 1 else 0) trainingData
    -- let grade = ((sum results) / (fromIntegral (length results))) * 100.0
    -- putStrLn $ "Res : " ++ show grade ++ "%"


    -- csvValid <- parseCSVOrError "/data/titanic/test.csv"
    -- let validationData = filterCSVValid csvValid
    -- print output
    -- let passengersId = [read (line !! 0) :: Int | line <- drop 1 csvValid, length line == 11]
    -- let output = zip passengersId calculated


    datas <- foldLoop [] 30 $ \currentDatas i -> do 
        putStrLn $ show i 
        validationData <- getData ((i-1)*10000)
        let res = map (forward model) validationData
        let calculated = [(asValue value :: [Float]) | value <- res]
        let sorted = map (\x -> reverse $ sortByFloat $ zip lookTable x) calculated
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
    --         let guess = asValue calculated :: [Float]
    --         let both = zip lookTable guess
    --         let sorted = reverse $ sortByFloat both
    --         print sorted
    --         return []
    return ()
    

lookTable = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

sortByFloat :: [(String, Float)] -> [(String, Float)]
sortByFloat = sortBy (\(_, x) (_, y) -> compare x y)


indexOfMax :: Ord a => [a] -> Int
indexOfMax xs = snd $ maximumBy (\x y -> compare (fst x) (fst y)) (zip xs [0..])
