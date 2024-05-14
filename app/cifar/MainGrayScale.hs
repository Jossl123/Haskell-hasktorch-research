-- PLEASE DOWNLOAD THE REQUIRED DATAS IF YOU WANT TO RUN THIS PROGRAM 
-- https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download

module MainGrayScale where

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

toGrayScaleList :: [Float] -> [Float]
toGrayScaleList [] = []
toGrayScaleList l = [(l !! 0 )* 0.2126 + (l !! 1) *  0.7152 + (l !! 2) * 0.7152] ++ (toGrayScaleList $ drop 3 l)

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
                pure (currentDatas ++ [(asTensor (toGrayScaleList rgbData), tensorOutput)])
    return datas

getData :: String -> IO [(Tensor, Tensor)]
getData folderName = do
    airplane <- getObjectData (folderName ++ "airplane") 0 10
    automobile <- getObjectData (folderName ++ "automobile") 1 10
    bird <- getObjectData (folderName ++ "bird") 2 10
    cat <- getObjectData (folderName ++ "cat") 3 10
    deer <- getObjectData (folderName ++ "deer") 4 10
    dog <- getObjectData (folderName ++ "dog") 5 10
    frog <- getObjectData (folderName ++ "frog") 6 10
    horse <- getObjectData (folderName ++ "horse") 7 10
    ship <- getObjectData (folderName ++ "ship") 8 10
    truck <- getObjectData (folderName ++ "truck") 9 10
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
        epochNb = 10000
        hypParams = MLPHypParams device 1024 [(64, Relu), (32, Relu), (10, Id)] -- Id | Sigmoid | Tanh | Relu | Elu | Selu

    trainingData <- getData "train/"
    validationData <- getData "test/"
    -- initModel <- sample hypParams
    initModel <- loadParams hypParams "models/trainingCifarGrayScale/cifar_750_30%_813loss.model"
    (trainedModel, _, losses) <- foldLoop (initModel, GD, []) epochNb $ \(model, opt, losses) i -> do 
        let epochLoss = sum (map (loss model) trainingData)
        let lossValue = asValue epochLoss :: Float
        putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue 
        (trainedModel, nOpt) <- runStep model opt epochLoss 0.0001
        when (i `mod` 50 == 0) $ do
            putStrLn "Saving..."
            let results = map (\(input, output) -> if (indexOfMax $ (asValue (forward model input) :: [Float])) == (indexOfMax $ (asValue output :: [Float])) then 1 else 0) validationData
            let grade = ((sum results) / (fromIntegral (length results))) * 100.0
            saveParams trainedModel ("models/trainingCifarGrayScale/cifar_" ++ show (i + 750) ++ "_" ++ show (round grade) ++ "%_" ++ show (round lossValue) ++ "loss.model" )
            drawLearningCurve "models/graph-cifar.png" "Learning Curve" [("", losses)]
            putStrLn "Saved!"
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
    --         let sorted = reverse $ sortBySnd both
    --         print sorted
            -- return []
    return ()
    

lookTable = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

sortBySnd :: [(String, Float)] -> [(String, Float)]
sortBySnd = sortBy (\(_, x) (_, y) -> compare x y)


indexOfMax :: Ord a => [a] -> Int
indexOfMax xs = snd $ maximumBy (\x y -> compare (fst x) (fst y)) (zip xs [0..])
