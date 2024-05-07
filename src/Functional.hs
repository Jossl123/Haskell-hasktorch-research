module Functional (softmax, accuracy, indexOfMax, precision, recall, f1, macroAvg, weightedAvg,sortByFloat) where
import Data.List            (sortBy, maximumBy)

import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.TensorFactories (zeros')
import Torch.Functional     (sumAll, exp, div, sub, add, mul)
import Torch.Layer.MLP      (MLPParams(..))

softmax :: Tensor -> Tensor
softmax input = Torch.Functional.div expInput expSum
    where expInput =  Torch.Functional.exp input
          expSum = sumAll expInput

outputResClean :: Tensor -> Tensor
outputResClean t = asTensor $ replicate imax (0.0 :: Float) ++ [1.0] ++ replicate (tLen - imax - 1) (0.0 :: Float)
    where imax = indexOfMax (asValue t :: [Float])
          tLen = length (asValue t :: [Float])

accuracy :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
accuracy model forward trainingData = (sum results) / (fromIntegral (length results))
    where results = map (\(input, output) -> if (indexOfMax $ (asValue (forward model input) :: [Float])) == (indexOfMax $ (asValue output :: [Float])) then 1 else 0) trainingData
       
getOneTpFnFp :: Tensor -> Tensor -> (Tensor,Tensor,Tensor)
getOneTpFnFp expected guess = result
    where expectedValue = asValue expected :: [Float]
          guessValue = asValue guess :: [Float] 
          nul = zeros' [(length (asValue expected :: [Float]))]
          result = if indexOfMax expectedValue == indexOfMax guessValue then (expected, nul, nul) else (nul, expected, guess)

getTpFnFp :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> (Tensor,Tensor,Tensor)
getTpFnFp model forward trainingData = (tp, fn, fp)
    where fTpFnFp = [ getOneTpFnFp output $ outputResClean $ forward model input | (input, output) <- trainingData]
          tp = foldl1 (Torch.Functional.add) [tps | (tps, _, _) <- fTpFnFp]
          fn = foldl1 (Torch.Functional.add) [fns | (_, fns, _) <- fTpFnFp]
          fp = foldl1 (Torch.Functional.add) [fps | (_, _, fps) <- fTpFnFp]

precision :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Tensor
precision model forward trainingData = Torch.Functional.div tp (Torch.Functional.add tp fp)
    where (tp, _, fp) = getTpFnFp model forward trainingData

recall :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Tensor
recall model forward trainingData = Torch.Functional.div tp (Torch.Functional.add tp fn)
    where (tp, fn, _) = getTpFnFp model forward trainingData

f1 :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Tensor
f1 model forward trainingData = Torch.Functional.div tp (Torch.Functional.add tp (Torch.Functional.div (Torch.Functional.add fn fp) 2.0))
    where (tp, fn, fp) = getTpFnFp model forward trainingData

macroAvg :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
macroAvg model forward trainingData = (asValue (sumAll f1score) :: Float) / (fromIntegral $ length (asValue f1score :: [Float]))
    where f1score = f1 model forward trainingData

weightedAvg :: MLPParams -> (MLPParams -> Tensor -> Tensor) -> [(Tensor,Tensor)] -> Float
weightedAvg model forward trainingData = result
    where expecteds = map snd trainingData
          weights = Torch.Functional.div (foldl1 (Torch.Functional.add) expecteds) (fromIntegral $ length trainingData) 
          f1score = f1 model forward trainingData
          weightedF1 = Torch.Functional.mul weights f1score
          result = asValue (sumAll weightedF1) :: Float

indexOfMax :: Ord a => [a] -> Int
indexOfMax xs = snd $ maximumBy (\x y -> compare (fst x) (fst y)) (zip xs [0..])


sortByFloat :: [(String, Float)] -> [(String, Float)]
sortByFloat = sortBy (\(_, x) (_, y) -> compare x y)
