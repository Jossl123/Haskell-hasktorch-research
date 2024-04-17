module Main where

import Torch.Functional     (mseLoss)
import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.NN             (sample)
import Torch.Optim          (GD(..), runStep, foldLoop)

import Torch.Train          (update)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Layer.Linear   (LinearHypParams(..), LinearParams(..), linearLayer)

trainingDatas :: [([Float], Float)]
trainingDatas = [([1, 1], 3),([2, 2], 6.1)]

createModel :: Device -> LinearHypParams
createModel device = LinearHypParams device True 2 1

forward :: LinearParams -> [Float] -> Tensor
forward model input = linearLayer model $ asTensor input 

loss :: LinearParams -> ([Float], Float) -> Tensor
loss model (input, output) = let y = forward model input 
                             in mseLoss y (asTensor output)

main :: IO()
main = do
    initialModel <- sample $ createModel device

    -- the foldLoop is the epoch loop
    (trainedModel, _) <- foldLoop (initialModel, optimizer) epochNb $ \(model,opt) i -> do 
        let epochLoss = sum (map (loss model) trainingDatas)
        putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show (asValue epochLoss :: Float)
        (trainedModel, nOpt) <- runStep model opt epochLoss 1e-3
        pure (trainedModel, nOpt)

    print trainedModel
    where device = Device CPU 0
          optimizer = GD
          epochNb = 100