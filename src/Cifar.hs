-- PLEASE DOWNLOAD THE REQUIRED DATAS IF YOU WANT TO RUN THIS PROGRAM 
-- https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download

module Cifar (cifar) where

import Torch.Optim          (foldLoop)
import ReadImage (imageToRGBList)

import Control.Monad (when)
import Data.List (sortBy, maximumBy)

import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.Functional     (mseLoss, Dim(..), exp, sumAll, div)
import Torch.NN             (sample)
import Torch.Optim          (GD(..), runStep, foldLoop)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP (MLPHypParams(..), MLPParams(..), ActName(..), mlpLayer)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools

getObjectData :: String -> Int -> Int -> IO [(Tensor, Tensor)]
getObjectData folderName output maxNb = do
    datas <- foldLoop [] 5000 $ \currentDatas i -> do 
        let fileName = show i
            paddingZero = concat $ take (4 - length fileName) ["0", "0", "0", "0"]
            imagePath = "data/cifar10/train/" ++ folderName ++ "/" ++ paddingZero ++ fileName ++ ".png"
        -- print imagePath
        image <- imageToRGBList imagePath
        case image of
            Left _ -> pure currentDatas
            Right rgbData -> do 
                let outputData = (take output (repeat 0 :: [Float])) ++ [1.0] ++ (take (maxNb - output - 1) (repeat 0 :: [Float]))
                let tensorOutput = asTensor outputData
                pure (currentDatas ++ [(asTensor rgbData, tensorOutput)])
    return datas

getTrainingData :: IO [(Tensor, Tensor)]
getTrainingData = do
    airplane <- getObjectData "airplane" 0 10
    automobile <- getObjectData "automobile" 1 10
    bird <- getObjectData "bird" 2 10
    cat <- getObjectData "cat" 3 10
    deer <- getObjectData "deer" 4 10
    dog <- getObjectData "dog" 5 10
    frog <- getObjectData "frog" 6 10
    horse <- getObjectData "horse" 7 10
    ship <- getObjectData "ship" 8 10
    truck <- getObjectData "truck" 9 10
    return $ concat $ [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
    

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
        epochNb = 1000000
        hypParams = MLPHypParams device 3072 [(64, Relu), (32, Relu), (10, Id)] -- Id | Sigmoid | Tanh | Relu | Elu | Selu

    trainingData <- getTrainingData
    initModel <- sample hypParams
    -- initModel <- loadParams hypParams "models/cifar_5000_524.7684.model"
    (trainedModel, _, losses) <- foldLoop (initModel, GD, []) epochNb $ \(model, opt, losses) i -> do 
        let epochLoss = sum (map (loss model) trainingData)
        let lossValue = asValue epochLoss :: Float
        putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue 
        (trainedModel, nOpt) <- runStep model opt epochLoss 0.0004
        when (i `mod` 500 == 0) $ do
            saveParams trainedModel ("models/cifar_" ++ show i ++ "_" ++ show lossValue ++ ".model" )
        pure (trainedModel, nOpt, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it 
    drawLearningCurve "models/graph-cifar.png" "Learning Curve" [("", losses)]
    saveParams trainedModel "models/cifar.model"



    model <- loadParams hypParams "models/cifar.model"
    
    let results = map (\(input, output) -> if (indexOfMax $ (asValue (forward model input) :: [Float])) == (indexOfMax $ (asValue output :: [Float])) then 1 else 0) trainingData
    let grade = ((sum results) / (fromIntegral (length results))) * 100.0
    putStrLn $ "Res : " ++ show grade ++ "%"

    -- image <- imageToRGBList "data/cifar10/train/deer/0139.png" 
    -- case image of
    --     Left _ -> return []
    --     Right rgbData -> do 
    --         let inputData = asTensor rgbData
    --         let calculated = forward model inputData
    --         let guess = asValue calculated :: [Float]
    --         let both = zip lookTable guess
    --         let sorted = reverse $ sortByFloat both
    --         print sorted
            -- return []

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
    

lookTable = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

sortByFloat :: [(String, Float)] -> [(String, Float)]
sortByFloat = sortBy (\(_, x) (_, y) -> compare x y)


indexOfMax :: Ord a => [a] -> Int
indexOfMax xs = snd $ maximumBy (\x y -> compare (fst x) (fst y)) (zip xs [0..])
