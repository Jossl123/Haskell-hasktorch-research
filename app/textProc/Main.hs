{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE StandaloneDeriving    #-}

module Main
  ( main
  ) where

import           Codec.Binary.UTF8.String (encode)
import qualified Data.ByteString.Lazy     as B
import          Data.Text.Lazy            as TL (unpack, pack)
import Data.Text.Lazy.Encoding            as TL (decodeUtf8, encodeUtf8)
import           Data.List                (nub)
import qualified Data.Map.Strict          as M
import           Data.Word                (Word8)
import           Data.Char                (toLower)  
import           GHC.Generics

import           Torch.Autograd           (makeIndependent, toDependent)
import           Torch.Functional         (embedding')
import           Torch.NN                 (Parameter, Parameterized (..))
import           Torch.Serialize          (loadParams, saveParams)
import           Torch.Tensor             (Tensor, asTensor, asValue)
import           Torch.TensorFactories    (eye', zeros')

import           Functional               (sortBySnd)

-- your text data (try small data first)
textFilePath = "data/textProc/review-texts.txt"

modelPath = "data/textProc/sample_embedding.params"

wordLstPath = "data/textProc/sample_wordlst.txt"

data EmbeddingSpec =
    EmbeddingSpec
        { wordNum :: Int -- the number of words
        , wordDim :: Int -- the dimention of word embeddings
        }
    deriving (Show, Eq, Generic)

data Embedding =
    Embedding
        { wordEmbedding :: Parameter
        }
    deriving (Show, Generic, Parameterized)

isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar str = str `elem` (map (head . encode)) [".", "!", "<", ">", "/", "\"", "-", "(", ")", ":", ";", ",", "?", "@"]

preprocess ::
    B.ByteString -- input
    -> [B.ByteString] -- wordlist per line
preprocess texts =  map (\s -> toByteString $ map toLower $ toString s) $ concat words
    where
        toByteString = TL.encodeUtf8 . TL.pack
        toString =  TL.unpack . TL.decodeUtf8  
        filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
        textLines     = B.split (head $ encode "\n") filteredtexts
        words         = map (B.split (head $ encode " ")) textLines

wordToIndexFactory ::
    [B.ByteString] -- wordlist
    -> (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd =
    M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0 ..]))

toyEmbedding :: EmbeddingSpec -> Tensor -- embedding
toyEmbedding EmbeddingSpec {..} = eye' wordNum wordDim

count :: Eq a => a -> [a] -> Int
count x = length . filter (x==)

main :: IO ()
main = do
    extractWordLst textFilePath
    wordlst <- loadWordLst wordLstPath
    let wordToIndex = wordToIndexFactory wordlst
    -- -- load params
    -- initWordEmb <- makeIndependent $ zeros' [1]
    -- let initEmb = Embedding {wordEmbedding = initWordEmb}
    -- loadedEmb <- loadParams initEmb modelPath
    -- let sampleTxt = B.pack $ encode "it"
    -- --     -- convert word to index
    --     idxes = map (map wordToIndex) (preprocess sampleTxt)
    -- --     -- convert to embedding
    --     embTxt = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor idxes)
    -- print idxes
    -- print embTxt
    -- print $ (toDependent $ wordEmbedding loadedEmb)
    return ()

loadWordLst :: FilePath -> IO [B.ByteString]
loadWordLst wordLstPath = do
    texts <- B.readFile wordLstPath
    let wordlst = B.split (head $ encode "\n") texts
    return wordlst

extractWordLst :: FilePath -> IO ()
extractWordLst textFilePath = do
    -- load text file
    texts <- B.readFile textFilePath
    putStrLn "loaded text file"
    -- create word lst (unique)
    let wordL = take 10000 $ preprocess texts
        wordFrequent = [ (word, count word wordL) | word <- (nub $ wordL)]
        wordFrequentSorted = reverse $ sortBySnd wordFrequent
        wordFrequentTop = take 1000 wordFrequentSorted
        wordlst = [word | (word, _) <- wordFrequentTop]
        wordToIndex = wordToIndexFactory wordlst
    putStrLn "created word list"
    -- create embedding(wordDim × wordNum)
    let embsddingSpec = EmbeddingSpec {wordNum = length wordlst, wordDim = length wordlst}
    wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
    let emb = Embedding {wordEmbedding = wordEmb}
    -- save params
    saveParams emb modelPath
  
    -- save word list
    B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
    