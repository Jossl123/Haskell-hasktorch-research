{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Sinus where

import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch
--------------------------------------------------------------------------------
-- MLP
--------------------------------------------------------------------------------

data MLPSpec = MLPSpec
  { feature_counts :: [Int],
    nonlinearitySpec :: Tensor -> Tensor
  }

data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) =
        scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize = 100

numIters = 300

sinus :: IO ()
sinus = do
  init <-
    sample $
      MLPSpec
        { feature_counts = [1, 256, 256, 1],
          nonlinearitySpec = Torch.elu'
        }
  trained <- foldLoop init numIters $ \state i -> do
    input <- randIO' [batchSize, 1] >>= return . (\x -> x * 6)
    let (y, y') = (tensorXOR input, squeezeAll $ mlp state input)
        loss = mseLoss y y'
    when (i `mod` 1 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    (newState, _) <- runStep state optimizer loss 1e-3
    return newState
  putStrLn "Final Model:"
  putStrLn $ "0 => " ++ (show $ squeezeAll $ mlp trained (asTensor [0 :: Float]))
  putStrLn $ "3.14=> " ++ (show $ squeezeAll $ mlp trained (asTensor [1.55 :: Float]))
  where
    optimizer = GD
    tensorXOR :: Tensor -> Tensor
    tensorXOR t = (Torch.sin a ) * 2.5 + 1.5
      where
        a = select 1 0 t
