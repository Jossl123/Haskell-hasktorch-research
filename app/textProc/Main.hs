-- PLEASE DOWNLOAD THE REQUIRED DATAS IF YOU WANT TO RUN THIS PROGRAM 
-- https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download
{-# LANGUAGE DeriveGeneric, RecordWildCards, MultiParamTypeClasses,FlexibleInstances  #-}

module Main where

import Functional           (sortBySnd)
import Torch.Model.Utils    (accuracy, f1, precision, recall, macroAvg, weightedAvg)
import Torch.Tensor.Util    (indexOfMax, oneHot')
import qualified Data.Text as T

import Graphics.Matplotlib
import System.Random
import Data.List.Split
import Data.List (elemIndex,foldl')
import Data.Maybe (fromJust)
import Torch.Optim          (foldLoop)
import ReadImage            (imageToRGBList)

import Control.Monad        (when,forM) --base
import Data.List            (sortBy, maximumBy)
import Torch.Functional      (matmul)

import Torch.Tensor         (asTensor, asValue, Tensor(..), select)
import Torch.TensorFactories (zeros')
import Torch.Functional     (mseLoss, binaryCrossEntropyLoss', Dim(..), exp, sumAll, div, squeezeAll, softmax, transpose2D)
import Torch.NN             (Parameterized,Randomizable,sample,flattenParameters)
import Torch.Optim          (GD(..), Adam(..), mkAdam, runStep, foldLoop)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Autograd       (toDependent)
import Torch.Layer.MLP      (MLPHypParams(..), MLPParams(..), ActName(..), mlpLayer)
import Torch.Layer.Linear   (LinearHypParams(..),LinearParams(..),linearLayer)
import ML.Exp.Chart         (drawLearningCurve,drawConfusionMatrix ) --nlp-tools
import Torch.Layer.NonLinear (ActName(..),decodeAct)
import Torch.Tensor.TensorFactories (oneAtPos)

import           Codec.Binary.UTF8.String (encode)
import Data.Maybe (catMaybes)
import qualified Data.ByteString.Lazy     as B
import           Data.Char                (toLower)
import           Data.List                (nub)
import qualified Data.Map.Strict          as M
import           Data.Text.Lazy           as TL (pack, unpack)
import           Data.Text.Lazy.Encoding  as TL (decodeUtf8, encodeUtf8)
import           Data.Word                (Word8)
import           GHC.Generics

import Debug.Trace          (trace)


-- | Example:
-- | toPairwise [(4,"a"),(5,"b"),(6,"c")] = [((4,"a"),(5,"b")),((5,"b"),(6,"c"))]
toPairwise :: [a] -> [(a,a)]
toPairwise [] = []
toPairwise [_] = []
toPairwise (x : (y : xs)) =
    scanl shift (x, y) xs
    where
        shift (_, p) q = (p, q)

data EmbeddingHypParams = EmbeddingHypParams {
    dev :: Device,
    wordDim :: Int,
    wordNum :: Int
    } deriving (Eq, Show)

-- | DeriveGeneric Pragmaが必要
data EmbeddingParams = EmbeddingParams {
    w1 :: LinearParams,
    w2 :: LinearParams
    } deriving (Generic)

instance Parameterized EmbeddingParams

instance Randomizable EmbeddingHypParams EmbeddingParams where
    sample EmbeddingHypParams{..} = do
        w1 <- sample $ LinearHypParams dev False wordNum wordDim
        w2 <- sample $ LinearHypParams dev False wordDim wordNum
        return $ EmbeddingParams w1 w2
        
-- as the input is a zero array with one value at 1, we can only mutliply the weight matrix linked to the 1 value
-- embLayerOpti :: EmbeddingParams -> Tensor -> Tensor 
-- embLayerOpti EmbeddingParams{..} input = softmax (Dim 0) $ l2
--     where
--         index = indexOfMax ( asValue input :: [Float] )
--         l1 = select 1 index (toDependent $ weight w1)
--         l2 = linearLayer w2 l1

embLayer :: EmbeddingParams -> Tensor -> Tensor 
embLayer EmbeddingParams{..} input = softmax (Dim 0) $ l2
    where l1 = linearLayer w1 input
          l2 = linearLayer w2 l1


loss :: EmbeddingParams -> (Tensor, Tensor) -> Tensor
loss model (input, output) = let y = embLayer model input
                             in mseLoss y output

-- forwardOpti :: MLPParams -> Tensor -> Tensor -- input is always a zero tensor with a value of 1 somewhere
-- forwardOpti model input = 
--     let word2vec = zip wordlst $ (asValue (toDependent $ weight $ fst $ head $ tail $ layers model) :: [[Float]])


main :: IO ()
main = do
    let device = Device CPU 0
        epochNb = 300
        wordDim = 16
        wordNum = 1000
        wordToReadInFile = 100000
        hypParams = EmbeddingHypParams device wordDim wordNum

    -- extractWordLst textFilePath wordToReadInFile wordNum
    wordlst <- loadWordLst wordLstPath
    trainingText <- B.readFile textFilePath

    model <- loadParams hypParams modelPath

    -- let trainingTextWords = take wordToReadInFile $ preprocess trainingText
    -- putStrLn "grabbing training data..."
    -- trainingData <- getTrainingData wordlst (take 10000 $ packOfFollowingWords trainingTextWords 1)
    -- initModel <- sample hypParams
    -- model <- trainModel initModel trainingData epochNb

    -- let word = asTensor $ fromJust $ M.lookup (TL.encodeUtf8 $ TL.pack "i'm") word2vecDict 
    let input = wordToOneHot (TL.encodeUtf8 $ TL.pack "i'm") wordlst
    let output = embLayer model input
    let outputWords = reverse $ sortBySnd $ zip wordlst $ (asValue output :: [Float])
    print $ take 10 outputWords

    let word2vec = zip wordlst $ (asValue (toDependent $ weight $ w2 model) :: [[Float]])
    let word2vecDict = M.fromList word2vec

    -- print word2vec

    let word = asTensor $ fromJust $ M.lookup (TL.encodeUtf8 $ TL.pack "software") word2vecDict 
    let res = asValue word :: [Float]

    -- let mostSim = take 10 $ reverse $ sortBySnd $ map (\(word, vec) -> (word, similarityCosine res vec)) word2vec
    print $ mostSimilar "sure" word2vec word2vecDict

    return ()

mostSimilar :: String -> [(B.ByteString, [Float])] -> M.Map B.ByteString [Float] -> [(B.ByteString, Float)]
mostSimilar word word2vec word2vecDict = 
    case M.lookup lazyWord word2vecDict of
        Nothing -> []
        Just vecWord -> take 10 $ reverse $ sortBySnd $ map (\(w, vec) -> (w, similarityCosine vecWord vec)) word2vec
    where lazyWord = TL.encodeUtf8 $ TL.pack word

trainModel :: EmbeddingParams -> [(Tensor, Tensor)] -> Int -> IO EmbeddingParams
trainModel model trainingData epochNb = do
    let optimizer = mkAdam 0 0.9 0.999 (flattenParameters model)
    (trainedModel, _, losses) <- foldLoop (model, optimizer, []) epochNb $ \(model, opt, losses) i -> do 
        let epochLoss = sum (map (loss model) trainingData)
        let lossValue = asValue epochLoss :: Float
        putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue 
        (trainedModel, nOpt) <- runStep model opt epochLoss 0.1
        pure (trainedModel, nOpt, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it 
    saveParams trainedModel modelPath
    return trainedModel


dotProduct :: [Float] -> [Float] -> Float
dotProduct x y = sum $ zipWith (*) x y

magnitude :: [Float] -> Float
magnitude x = sqrt $ sum $ map (^2) x

similarityCosine :: [Float] -> [Float] -> Float
similarityCosine x y = dotProduct x y / (magnitude x * magnitude y)

wordToIndex :: B.ByteString -> [B.ByteString] -> Int
wordToIndex word wordlst = if length indexs > 0 then head indexs else -1
    where indexs = (catMaybes [elemIndex word wordlst])

wordToOneHot :: B.ByteString -> [B.ByteString] -> Tensor
wordToOneHot word wordlst = if index >= 0 then oneAtPos index (length wordlst) else zeros' [length wordlst]
    where index = wordToIndex word wordlst

-- your text data (try small data first)
textFilePath = "data/textProc/review-texts.txt"

modelPath = "data/textProc/sample_embedding.params"

wordLstPath = "data/textProc/sample_wordlst.txt"


isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar str =
    str `elem`
    (map (head . encode))
        [".", "!", "<", ">", "/", "\"", "(", ")", ":", ";", ",", "-", "?", "@", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

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

count :: Eq a => a -> [a] -> Int
count x = length . filter (x ==)

packOfFollowingWords :: [B.ByteString] -> Int -> [([B.ByteString], B.ByteString)] -- check if length of wordlst is greater than n
packOfFollowingWords [] _ = []
packOfFollowingWords [x] _ = []
packOfFollowingWords (x:xs) n = if length xs > (n+1) then ([p2], p1) : ([p3], p1) : ([p1], p2) : ([p1], p3) : packOfFollowingWords xs n else []
    where p1 = head xs
          p2 = head $ tail xs
          p3 = head $ tail $ tail xs

-- Convert data into training data
getTrainingData :: [B.ByteString] -> [([B.ByteString], B.ByteString)] -> IO [(Tensor, Tensor)]
getTrainingData wordlst dataPack = do 
    let input x = asTensor $ concat $ map (\word -> asValue (oneAtPos word (length wordlst)) :: [Float]) (idxLst x)
        idxLst x = catMaybes $ map (`elemIndex` wordlst) x
        output y = asTensor $ concat $ map (\word -> asValue (oneAtPos word (length wordlst)) :: [Float]) (catMaybes [elemIndex y wordlst])
        res = map (\(x, y) -> (input x, output y)) dataPack
        filteredRes = filter (\(x, y) -> (length (asValue x :: [Float]) > 0) && (length (asValue y :: [Float]) > 0)) res
    return filteredRes

loadWordLst :: FilePath -> IO [B.ByteString]
loadWordLst wordLstPath = do
    texts <- B.readFile wordLstPath
    let wordlst = B.split (head $ encode "\n") texts
    return wordlst

extractWordLst :: FilePath -> Int -> Int -> IO ()
extractWordLst textFilePath wordsToReadInFile wordNbToGrab = do
    -- load text file
    texts <- B.readFile textFilePath
    putStrLn "loaded text file"
        -- create word lst (unique)
    let wordL = take wordsToReadInFile $ preprocess texts
        wordFrequent = [(word, count word wordL) | word <- (nub $ wordL)]
        wordFrequentSorted = reverse $ sortBySnd wordFrequent
        wordFrequentTop = take wordNbToGrab wordFrequentSorted
        wordlst = [word | (word, _) <- wordFrequentTop]
        wordToIndex = wordToIndexFactory wordlst
    putStrLn "created word list"
        
    -- save word list
    B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
    putStrLn "saved word list"
