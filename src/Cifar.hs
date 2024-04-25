-- PLEASE DOWNLOAD THE REQUIRED DATAS IF YOU WANT TO RUN THIS PROGRAM 
-- https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download

module Cifar (cifar) where

import Torch.Optim          (foldLoop)
import ReadImage (imageToRGBList)

getObjectData :: String -> Int -> Int -> IO [([Float], [Float])]
getObjectData folderName output maxNb = do
    (datas) <- foldLoop ([]) 3 $ \(currentDatas) i -> do 
        let fileName = show i
        let paddingZero = concat $ take (4-(length fileName)) ["0","0","0","0"]
        print $ paddingZero ++ fileName
        image <- imageToRGBList ("data/cifar10/train/"++ folderName ++"/"++ paddingZero ++ fileName ++ ".png")
        case image of
            Left _ -> pure (currentDatas)
            Right rgbData -> pure (currentDatas ++ [(rgbData, (take output [0,0..]) ++ [1] ++ (take (maxNb - output - 1) [0,0..]) )])
    return datas

getTrainingData :: IO [([Float], [Float])]
getTrainingData = do
    -- airplane <- getObjectData "airplane" 0 10
    automobile <- getObjectData "automobile" 1 10
    bird <- getObjectData "bird" 2 10
    -- cat <- getObjectData "cat" 3 10
    -- deer <- getObjectData "deer" 4 10
    return $ concat $ [automobile, bird]
    
cifar :: IO ()
cifar = do
    trainingData <- getTrainingData
    print trainingData