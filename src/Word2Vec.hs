{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards       #-}

module Word2Vec (EmbeddingHypParams(..),EmbeddingParams(..), embLayer, similarityCosine) where

import           GHC.Generics
import           Torch.NN
import           Torch.Device
import           Torch.Layer.Linear           (LinearHypParams (..), LinearParams (..), linearLayer)
import           Torch.Functional             (Dim (..), softmax)
import           Torch.Tensor                 (Tensor (..))

-- embLayerOpti EmbeddingParams{..} input = softmax (Dim 0) $ l2
--     where
--         index = indexOfMax ( asValue input :: [Float] )
--         l1 = select 1 index (toDependent $ weight w1)
--         l2 = linearLayer w2 l1
embLayer :: EmbeddingParams -> Tensor -> Tensor
embLayer EmbeddingParams {..} input = softmax (Dim 0) $ l2
    where
        l1 = linearLayer w1 input
        l2 = linearLayer w2 l1

data EmbeddingHypParams =
    EmbeddingHypParams
        { dev     :: Device
        , wordDim :: Int
        , wordNum :: Int
        }
    deriving (Eq, Show)

data EmbeddingParams =
  EmbeddingParams
        { w1 :: LinearParams
        , w2 :: LinearParams
        }
    deriving (Show, Generic)

instance Parameterized EmbeddingParams

instance Randomizable EmbeddingHypParams EmbeddingParams
    where
    sample EmbeddingHypParams {..} = do
        w1 <- sample $ LinearHypParams dev False wordNum wordDim
        w2 <- sample $ LinearHypParams dev False wordDim wordNum
        return $ EmbeddingParams w1 w2


dotProduct :: [Float] -> [Float] -> Float
dotProduct x y = sum $ zipWith (*) x y

magnitude :: [Float] -> Float
magnitude x = sqrt $ sum $ map (^ 2) x

similarityCosine :: [Float] -> [Float] -> Float
similarityCosine x y = dotProduct x y / (magnitude x * magnitude y)
