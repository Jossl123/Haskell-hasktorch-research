-- PLEASE DOWNLOAD THE REQUIRED DATAS IF YOU WANT TO RUN THIS PROGRAM
-- https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards       #-}

module Main where

import qualified Data.Text                    as T
import           Functional                   (sortBySnd)
import           Torch.Model.Utils            (accuracy, f1, macroAvg,
                                               precision, recall, weightedAvg)
import           Torch.Tensor.Util            (indexOfMax, oneHot')

import           Data.List                    (elemIndex, foldl')
import           Data.List.Split
import           Data.Maybe                   (fromJust)
import           Data.Time.Clock              (diffUTCTime, getCurrentTime)
import           Graphics.Matplotlib
import           ReadImage                    (imageToRGBList)
import           System.Random
import           Torch.Optim                  (foldLoop)

import           Control.Monad                (forM, when)
import           Data.List                    (maximumBy, sortBy)
import           Torch.Functional             (matmul)

import           ML.Exp.Chart                 (drawConfusionMatrix,
                                               drawLearningCurve)
import           Torch.Autograd               (toDependent)
import           Torch.Device                 (Device (..), DeviceType (..))
import           Torch.Functional             (Dim (..),
                                               binaryCrossEntropyLoss', div,
                                               exp, mseLoss, softmax,
                                               squeezeAll, sumAll, transpose2D)
import           Torch.Layer.Linear           (LinearHypParams (..),
                                               LinearParams (..), linearLayer)
import           Torch.Layer.MLP              (ActName (..), MLPHypParams (..),
                                               MLPParams (..), mlpLayer)
import           Torch.Layer.NonLinear        (ActName (..), decodeAct)
import           Torch.NN                     (Parameterized, Randomizable,
                                               flattenParameters, sample)
import           Torch.Optim                  (Adam (..), GD (..), foldLoop,
                                               mkAdam, runStep)
import           Torch.Tensor                 (Tensor (..), asTensor, asValue,
                                               select, shape, toInt)
import           Torch.Tensor.TensorFactories (oneAtPos)
import           Torch.TensorFactories        (zeros')
import           Torch.Train                  (loadParams, saveParams, update)

import           Codec.Binary.UTF8.String     (encode)
import qualified Data.ByteString.Lazy         as B
import           Data.Char                    (toLower)
import           Data.List                    (nub)
import qualified Data.Map.Strict              as M
import           Data.Maybe                   (catMaybes)
import           Data.Text.Lazy               as TL (pack, unpack)
import           Data.Text.Lazy.Encoding      as TL (decodeUtf8, encodeUtf8)
import           Data.Word                    (Word8)
import           GHC.Generics

import           Debug.Trace                  (trace)

-- | Example:
-- | toPairwise [(4,"a"),(5,"b"),(6,"c")] = [((4,"a"),(5,"b")),((5,"b"),(6,"c"))]
toPairwise :: [a] -> [(a, a)]
toPairwise [] = []
toPairwise [_] = []
toPairwise (x:(y:xs)) = scanl shift (x, y) xs
    where
        shift (_, p) q = (p, q)

data EmbeddingHypParams =
    EmbeddingHypParams
        { dev     :: Device
        , wordDim :: Int
        , wordNum :: Int
        }
    deriving (Eq, Show)


-- | DeriveGeneric Pragmaが必要
data EmbeddingParams =
  EmbeddingParams
        { w1 :: LinearParams
        , w2 :: LinearParams
        }
    deriving (Generic)

instance Parameterized EmbeddingParams

instance Randomizable EmbeddingHypParams EmbeddingParams
    where
    sample EmbeddingHypParams {..} = do
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
embLayer EmbeddingParams {..} input = softmax (Dim 0) $ l2
    where
        l1 = linearLayer w1 input
        l2 = linearLayer w2 l1

loss :: EmbeddingParams -> (Tensor, Tensor) -> Tensor
loss model (input, output) =
    let y = embLayer model input
    in mseLoss y output

chronometer :: IO a -> IO (a, Double)
chronometer action = do
    start <- getCurrentTime
    result <- action
    end <- getCurrentTime
    let diff = realToFrac $ diffUTCTime end start
    return (result, diff)

textFilePath = "data/textProc/review-texts.txt"

modelPath = "data/textProc/sample_embedding.params"

wordLstPath = "data/textProc/sample_wordlst.txt"

main :: IO ()
main = do
    let device = Device CPU 0
        epochNb = 20
        wordDim = 16
        wordNum = 1000
        wordToReadInFile = 1000000
        hypParams = EmbeddingHypParams device wordDim wordNum
    ----------------- DATA PREPROCESSING -----------------
    -- extractWordLst textFilePath wordToReadInFile wordNum
    wordlst <- loadWordLst wordLstPath
    exportTrainingData
        "data/textProc/training_data.txt"
        textFilePath
        wordlst
        wordToReadInFile
        1000
    ----------------- TRAINING -----------------
    -- trainingData <- loadTrainingData "data/textProc/training_data.txt" wordlst
    -- initModel <- sample hypParams
    -- model <- trainModel initModel trainingData epochNb
    ----------------- TESTING -----------------
    model <- loadParams hypParams modelPath
    let word2vec = zip wordlst $ (asValue (toDependent $ weight $ w2 model) :: [[Float]])
    let word2vecDict = M.fromList word2vec
    -- let wordVec = asTensor $ fromJust $ M.lookup (TL.encodeUtf8 $ TL.pack "software") word2vecDict
    -- let res = asValue word :: [Float]
    print $ mostSimilar "game" word2vec word2vecDict
    return ()

exportTrainingData :: FilePath -> FilePath -> [B.ByteString] -> Int -> Int -> IO ()
exportTrainingData filePath trainingText wordlst wordToReadInFile traininSetSize = do
    putStrLn "grabbing training data..."
    trainingText <- B.readFile textFilePath
    let trainingTextWords = take wordToReadInFile $ preprocess trainingText
        trainingDataPack =
            take traininSetSize $ packOfFollowingWords trainingTextWords
        spacedTrainingData =
            map
            (\(bs1, bs2) -> bs1 `B.append` (B.pack $ encode " ") `B.append` bs2)
            trainingDataPack
    B.writeFile filePath (B.intercalate (B.pack $ encode "\n") spacedTrainingData)
    putStrLn "training data saved"

loadTrainingData :: FilePath -> [B.ByteString] -> IO [(Tensor, Tensor)]
loadTrainingData trainingDataPath wordlst = do
    trainingDataWords <- B.readFile trainingDataPath
    let trainingDataw =
            map (\x -> (head x, head $ tail x)) $
            map (\bs -> B.split (head $ encode " ") bs) $
            B.split (head $ encode "\n") trainingDataWords
    trainingData <- getTrainingData wordlst trainingDataw
    return trainingData


-- | Find the most similar words to a given word
mostSimilar ::
    String
    -> [(B.ByteString, [Float])]
    -> M.Map B.ByteString [Float]
    -> [(B.ByteString, Float)]
mostSimilar word word2vec word2vecDict =
    case M.lookup lazyWord word2vecDict of
        Nothing -> []
        Just vecWord ->
            take 10 $ 
            reverse $
            sortBySnd $ map (\(w, vec) -> (w, similarityCosine vecWord vec)) word2vec
    where
        lazyWord = TL.encodeUtf8 $ TL.pack word

trainModel :: EmbeddingParams -> [(Tensor, Tensor)] -> Int -> IO EmbeddingParams
trainModel model trainingData epochNb = do
    let optimizer = mkAdam 0 0.9 0.999 (flattenParameters model)
    (trainedModel, _, losses) <-
        foldLoop (model, optimizer, []) epochNb $ \(model, opt, losses) i -> do
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
magnitude x = sqrt $ sum $ map (^ 2) x

similarityCosine :: [Float] -> [Float] -> Float
similarityCosine x y = dotProduct x y / (magnitude x * magnitude y)

wordToOneHot :: B.ByteString -> [B.ByteString] -> Tensor
wordToOneHot word wordlst =
    if index < (length wordlst) - 1
        then oneAtPos index (length wordlst)
        else zeros' [length wordlst]
    where
        wordToIndex = wordToIndexFactory wordlst
        index       = wordToIndex word

isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar str =
    str `elem` (map (head . encode))
        [ ".", "!", "<", ">", "/", "\"", "(", ")", ":", ";", ",", "-", "?", "@", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

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

packOfFollowingWords :: [B.ByteString] -> [(B.ByteString, B.ByteString)] -- check if length of wordlst is greater than n
packOfFollowingWords [] = []
packOfFollowingWords [x] = []
packOfFollowingWords (x:xs) =
    if length xs > 2
        then (p2, p1) : (p3, p1) : (p1, p2) : (p1, p3) : packOfFollowingWords xs
        else []
    where
        p1 = head xs
        p2 = head $ tail xs
        p3 = head $ tail $ tail xs

-- Convert data into training data
getTrainingData :: [B.ByteString] -> [(B.ByteString, B.ByteString)] -> IO [(Tensor, Tensor)]
getTrainingData wordlst dataPack = do
    let wordToIndex = wordToIndexFactory wordlst
        input x = wordToOneHot x wordlst
        output y = wordToOneHot y wordlst
        res = map (\(x, y) -> (input x, output y)) dataPack
        filteredRes =
            filter
            (\(x, y) -> ((toInt $ sumAll x) > 0) && (toInt $ sumAll y) > 0)
            res
    return filteredRes

loadWordLst :: FilePath -> IO [B.ByteString]
loadWordLst wordLstPath = do
    texts <- B.readFile wordLstPath
    let wordlst = B.split (head $ encode "\n") texts
    return wordlst

extractWordLst :: FilePath -> Int -> Int -> IO ()
extractWordLst textFilePath wordsToReadInFile wordNbToGrab
    -- load text file
    = do
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
