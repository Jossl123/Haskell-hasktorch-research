{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards       #-}

module Word2Vec (EmbeddingHypParams(..),EmbeddingParams(..), embLayer, similarityCosine, wordToOneHotLookupFactory, wordToOneHotFactory, wordToIndexFactory, preprocess) where

import           GHC.Generics
import           Torch.NN
import           Torch.Device
import           Torch.Layer.Linear           (LinearHypParams (..), LinearParams (..), linearLayer)
import           Torch.Functional             (Dim (..), softmax, embedding', stack)
import           Torch.Tensor                 (Tensor (..), asTensor)
import qualified Data.Map.Strict              as M
import qualified Data.ByteString.Lazy         as B
import           Torch.Tensor.TensorFactories (oneAtPos)
import           Data.Text.Lazy               as TL (pack, unpack)
import           Data.Text.Lazy.Encoding      as TL (decodeUtf8, encodeUtf8)
import           Codec.Binary.UTF8.String     (encode)
import           TextTreatement               (replaceUnnecessaryChar)
import           Data.Char                    (toLower)






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

wordToOneHotLookupFactory :: Int -> Tensor
wordToOneHotLookupFactory wordNum = stack (Dim 0) $ map (\i -> oneAtPos i wordNum) [0..(wordNum - 1)]

-- 100000 in [125, 123] sec
wordToOneHotFactory ::  (B.ByteString -> Int) -> Tensor -> (B.ByteString -> Tensor)
wordToOneHotFactory wordToIndex wordToOneHotLookup word = embedding' wordToOneHotLookup (asTensor [[wordToIndex word :: Int]])

wordToIndexFactory ::
    [B.ByteString] -- wordlist
    -> (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd =
    M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0 ..]))


preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (map (\s -> toByteString $ map toLower $ toString s)) words
    where
        toByteString = TL.encodeUtf8 . TL.pack
        toString     = TL.unpack . TL.decodeUtf8
        replacedTexts = B.pack $ map replaceUnnecessaryChar (B.unpack texts)
        textLines     = B.split (head $ encode "\n") replacedTexts
        wordsLst      = map (B.split (head $ encode " ")) textLines
        words         = map (filter (not . B.null)) $ wordsLst
