-- https://jaketae.github.io/study/word2vec/

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
import           Data.Char                (toLower)
import           Data.List                (nub)
import qualified Data.Map.Strict          as M
import           Data.Text.Lazy           as TL (pack, unpack)
import           Data.Text.Lazy.Encoding  as TL (decodeUtf8, encodeUtf8)
import           Data.Word                (Word8)
import           GHC.Generics

import           Torch.Autograd           (makeIndependent, toDependent)
import           Torch.Functional         (embedding', mseLoss, transpose2D)
import           Torch.NN                 (Parameter, Parameterized (..))
import           Torch.Serialize          (loadParams, saveParams)
import           Torch.Tensor             (Tensor, asTensor, asValue)
import           Torch.Optim              (GD(..), Adam(..), mkAdam, runStep, foldLoop)
import           Torch.TensorFactories    (eye', zeros')
import           Torch.Device             (Device(..), DeviceType(..))
import           Torch.NN                 (sample)
import           Torch.Layer.MLP          (MLPHypParams(..), MLPParams(..), ActName(..))
import Torch.Train (update)

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
        { wordEmbedding :: Parameter,
          w2 :: Parameter
        }
    deriving (Show, Generic, Parameterized)



mlpLayer :: Embedding -> Tensor -> Tensor
mlpLayer Embedding {..} input = a1
    where a1 = (toDependent wordEmbedding) * (transpose2D input)
        --   a2 = (toDependent w2) * a1

tro :: Embedding -> IO () -- print word embedding
tro Embedding {..} = do
    let wordEmb = asValue (toDependent $ wordEmbedding ):: [[Float]]
    print wordEmb

isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar str =
    str `elem`
    (map (head . encode))
        [".", "!", "<", ">", "/", "\"", "-", "(", ")", ":", ";", ",", "?", "@"]

preprocess ::
     B.ByteString -- input
  -> [B.ByteString] -- wordlist per line
preprocess texts = map (\s -> toByteString $ map toLower $ toString s) words
    where
        toByteString  = TL.encodeUtf8 . TL.pack
        toString      = TL.unpack . TL.decodeUtf8
        filteredtexts = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
        textLines     = B.split (head $ encode "\n") filteredtexts
        wordsLst      = map (B.split (head $ encode " ")) textLines
        words         = filter (not . B.null) $ concat wordsLst

wordToIndexFactory ::
     [B.ByteString] -- wordlist
  -> (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd =
    M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0 ..]))

toyEmbedding :: EmbeddingSpec -> Tensor -- embedding
toyEmbedding EmbeddingSpec {..} = eye' wordNum wordDim

count :: Eq a => a -> [a] -> Int
count x = length . filter (x ==)


packOfFollowingWords :: [B.ByteString] -> Int -> [([B.ByteString], B.ByteString)]
packOfFollowingWords [] _ = [] 
packOfFollowingWords [x] _ = []
packOfFollowingWords (x:xs) n = (take n $ tail xs , head xs) : packOfFollowingWords xs n 


getTrainingData :: ([Int] -> Tensor) -> (B.ByteString -> Int) -> [([B.ByteString], B.ByteString)] -> [(Tensor, Tensor)]
getTrainingData embdFunc wordToIndex dataPack = map (\(x, y) -> (flatTensor $ input x, flatTensor $ output y)) dataPack
    where input x = embdFunc (map wordToIndex x)
          output x  = embdFunc [wordToIndex x]
          flatTensor x = asTensor $ concat $ (asValue x :: [[Float]])

embd :: Embedding -> [Int] -> Tensor
embd loadedEmb idxes = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor idxes)

main :: IO ()
main = do
    extractWordLst textFilePath
    wordlst <- loadWordLst wordLstPath
    let wordToIndex = wordToIndexFactory wordlst
        device = Device CPU 0
        epochNb = 7
        hypParams = MLPHypParams device 20 [(5, Id), (10, Softmax)] -- Id | Sigmoid | Tanh | Relu | Elu | Selu

    -- extractWordLst textFilePath
    -- load params
    initWordEmb <- makeIndependent $ zeros' [1]
    let initEmb = Embedding {wordEmbedding = initWordEmb, w2 = initWordEmb}
    loadedEmb <- loadParams initEmb modelPath
    let fwd = mlpLayer loadedEmb (asTensor ([0,0,1,0,0,0,0,0,0,0] :: [Int]))
    print (asTensor ([[0,0,1,0,0,0,0,0,0,0]] :: [[Int]]))
    print loadedEmb
    print fwd
    -- let optimizer = mkAdam 0 0.9 0.999 (flattenParameters loadedEmb)
    -- (trainedModel, nOpt) <- update loadedEmb optimizer 2.3 0.002

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
    let wordL = take 1000 $ preprocess texts
        wordFrequent = [(word, count word wordL) | word <- (nub $ wordL)]
        wordFrequentSorted = reverse $ sortBySnd wordFrequent
        wordFrequentTop = take 10 wordFrequentSorted
        wordlst = [word | (word, _) <- wordFrequentTop]
        wordToIndex = wordToIndexFactory wordlst
    putStrLn "created word list"
        -- create embedding(wordDim × wordNum)
    let embsddingSpec =
            EmbeddingSpec {wordNum = length wordlst, wordDim = 5}
    let embsddingSpec2 =
            EmbeddingSpec {wordNum = 5, wordDim = length wordlst}
    wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
    wordEmb2 <- makeIndependent $ toyEmbedding embsddingSpec2
    let emb = Embedding {wordEmbedding = wordEmb, w2 = wordEmb2}
        -- save params
    saveParams emb modelPath
        
    -- save word list
    B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
    putStrLn "saved word list"
