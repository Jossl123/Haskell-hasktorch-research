module Functional (sortBySnd) where
import Data.List            (sortBy, maximumBy)

import Graphics.Matplotlib
import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.TensorFactories (zeros')
import Torch.Tensor.TensorFactories (oneAtPos2d)
import Torch.Functional     (sumAll, exp, div, sub, add, mul)
import Torch.Layer.MLP      (MLPParams(..))

sortBySnd :: (Ord c) => [(a, c)] -> [(a, c)]
sortBySnd = sortBy (\(_, x) (_, y) -> compare x y)
