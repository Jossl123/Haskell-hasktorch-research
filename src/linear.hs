module Main where

import Control.Monad (when)
import Torch

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul (matmul t weight + bias) weight2 + bias2
  where
    weight = asTensor ([2] :: [Float])
    bias = full' [3] (3.14 :: Float)
    weight2 = asTensor ([42.0, 64.0, 96.0] :: [Float])
    bias2 = full' [1] (3.14 :: Float)

model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

main :: IO ()
main = do
    init <- sample $ LinearSpec{in_features = numFeatures, out_features = 1}
    randGen <- mkGenerator (Device CPU 0) 12345
    (trained, _) <- foldLoop (init, randGen) 2000 $ \(state, randGen) i -> do
        let (input, randGen') = randn' [batchSize, numFeatures] randGen
            (y, y') = (groundTruth input, model state input)
            loss = mseLoss y y'
        when (i `mod` 100 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
        (state', _) <- runStep state GD loss 5e-3
        pure (state', randGen')
    pure ()
  where
    batchSize = 3
    numFeatures = 1
