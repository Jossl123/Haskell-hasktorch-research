module MyLinear where

import Torch.Functional     (mseLoss)
import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.NN             (sample)
import Torch.Optim          (GD(..), runStep, foldLoop)

import Torch.Train          (update)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Layer.Linear   (LinearHypParams(..), LinearParams(..), linearLayer)

trainingDatas :: [([Float], Float)]
trainingDatas = [([1, 1], 2.8),([2, 2], 6.1),([3, 3], 8.9),([2, 1.9], 6.1)]

validationDatas :: [([Float], Float)]
validationDatas = [([4,4], 12),([2, 2], 6),([3, 3], 9),([6,7], 15.1)]

createModel :: Device -> LinearHypParams
createModel device = LinearHypParams device True 2 1

forward :: LinearParams -> [Float] -> Tensor
forward model input = linearLayer model $ asTensor input 

loss :: LinearParams -> ([Float], Float) -> Tensor
loss model (input, output) = let y = forward model input 
                             in mseLoss y (asTensor output)

linear :: IO()
linear = do
    -- sample initiate a network with random value with the size of our model
    initialModel <- sample $ createModel device

    -- the foldLoop is the epoch loop
    (trainedModel, _) <- foldLoop (initialModel, optimizer) epochNb $ \(model,opt) i -> do 
        let epochLoss = sum (map (loss model) trainingDatas)
        putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show (asValue epochLoss :: Float)
        (trainedModel, nOpt) <- runStep model opt epochLoss 1e-3
        pure (trainedModel, nOpt) -- pure : transform return type to IO because foldLoop need it 
    print trainedModel

    ----- VALIDATION ----- 
    let validationOutputs = map (\(input, _) -> asValue (forward trainedModel input) :: Float) validationDatas

    -- zip expected and actual values to display
    putStrLn $ "(Expected, Actual) : " ++ show (asTensor $ map tupleToList2 $ zip validationOutputs (map snd validationDatas))
    let mse = sum $ map (loss trainedModel) validationDatas
    putStrLn $ "MSE : " ++ show (asValue mse :: Float)

    where device = Device CPU 0
          optimizer = GD
          epochNb = 100


-- convert tuple to list (not needed for the neural network, only for display purposes)
tupleToList2 :: (a, a) -> [a]
tupleToList2 (x, y) = [x, y]