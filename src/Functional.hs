module Functional (softmax, accuracy, indexOfMax) where
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
import ML.Exp.Chart         (drawLearningCurve) --nlp-tools

softmax :: Tensor -> Tensor
softmax input = 
    let expInput =  Torch.Functional.exp input
        expSum = sumAll expInput
    in Torch.Functional.div expInput expSum

accuracy :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
accuracy model forward trainingData = (sum results) / (fromIntegral (length results))
    where results = map (\(input, output) -> if (indexOfMax $ (asValue (forward model input) :: [Float])) == (indexOfMax $ (asValue output :: [Float])) then 1 else 0) trainingData

precision :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
precision model forward trainingData = 0.2

recall :: Float 
recall = 0.2

indexOfMax :: Ord a => [a] -> Int
indexOfMax xs = snd $ maximumBy (\x y -> compare (fst x) (fst y)) (zip xs [0..])
