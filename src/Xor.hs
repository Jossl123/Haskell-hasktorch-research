module Xor (xor) where
    
import Control.Monad (forM_)        --base

import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.Functional     (mseLoss)
import Torch.NN             (sample)
import Torch.Optim          (GD(..), foldLoop)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP (MLPHypParams(..), MLPParams(..), ActName(..), mlpLayer)

loss :: MLPParams -> ([Float], Float) -> Tensor
loss model (input, output) = let y = mlpLayer model $ asTensor input 
                             in mseLoss y (asTensor output)

trainingData :: [([Float], Float)]
trainingData = take 12 $ cycle [([0,0],0),([1,0],1),([0,1],1),([1,1],0)]

xor :: IO ()
xor = do
    let device = Device CPU 0
        epochNb = 2000
        hypParams = MLPHypParams device 2 [(3, Tanh),(3, Tanh), (1, Sigmoid)]
    initModel <- sample hypParams
    (trainedModel, _, losses) <- foldLoop (initModel, GD, []) epochNb $ \(model, opt, losses) i -> do 
        let epochLoss = sum (map (loss model) trainingData)
        let lossValue = asValue epochLoss :: Float
        putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue 
        (trainedModel, nOpt) <- update model opt epochLoss 0.1
        pure (trainedModel, nOpt, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it 
    forM_ ([[1,1],[1,0],[0,1],[0,0]::[Float]]) $ \input -> do
        putStr $ show $ input
        putStr ": "
        putStrLn $ show ((mlpLayer trainedModel $ asTensor input))
    return ()
    