
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE BangPatterns #-}


module Main where

import qualified Data.Text                    as T
import           Functional                   (sortBySnd)

import Data.Ord (comparing)
import           Data.List                    (foldl')
import System.Random.Shuffle (shuffle')
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
import           Torch.Functional             (Dim (..), embedding', stack,
                                               binaryCrossEntropyLoss', div,
                                               exp, mseLoss, softmax, sub, add, log, mul, min,
                                               squeezeAll, sumAll, transpose2D)
import           Torch.Layer.Linear           (LinearHypParams (..),
                                               LinearParams (..), linearLayer)
import           Torch.Layer.MLP              (ActName (..), MLPHypParams (..),
                                               MLPParams (..), mlpLayer)
import           Torch.Layer.NonLinear        (ActName (..), decodeAct)
import           Torch.NN                     (Parameterized, Randomizable,
                                               flattenParameters, sample)
import           Torch.Optim                  (Adam (..), GD (..), foldLoop,
                                               mkAdam, runStep, grad')
import           Torch.Tensor                 (Tensor (..), asTensor, asValue,
                                               select, shape, toInt)
import           Torch.Tensor.TensorFactories (oneAtPos)
import           Torch.TensorFactories        (zeros')
import           Torch.Train                  (loadParams, saveParams)

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


import Word2Vec (EmbeddingHypParams (..), EmbeddingParams (..), embLayer, similarityCosine)
import Utils (chronometer)
import TextTreatement (replaceUnnecessaryChar)
import Debug.Trace (trace)

loss :: EmbeddingParams -> (Tensor, Tensor) -> Tensor
loss model (input, output) =
    let y = embLayer model input
        epsilon = asTensor (1e-8 :: Float)
        logp = Torch.Functional.log $ Torch.Functional.add epsilon y
        sumA = sumAll $ Torch.Functional.mul logp output
    in Torch.Functional.mul (asTensor [-1 :: Float]) sumA

textFilePath = "data/textProc/review-texts.txt"
modelPath = "data/textProc/sample_embedding.params"
wordLstPath = "data/textProc/sample_wordlst.txt"

main :: IO ()
main = do
    let device = Device CPU 0
        epochNb = 100
        wordDim = 16
        wordNum = 1000
        wordToReadInFile = 100000
        hypParams = EmbeddingHypParams device wordDim wordNum
    ----------------- DATA PREPROCESSING -----------------
    -- (wordlst, extractDuration) <- chronometer $ extractWordLst textFilePath wordToReadInFile wordNum
    -- putStrLn $ "Word list extraction duration: " ++ show extractDuration ++ "s"
    wordlst <- loadWordLst wordLstPath
    let wordToOneHot = wordToOneHotFactory  (wordToIndexFactory wordlst) (wordToOneHotLookupFactory wordNum)
        testWord = TL.encodeUtf8 $ TL.pack "phone"
    -- print $ wordToOneHot testWord

    -- (_, duration) <- chronometer $ exportTrainingData
    --     "data/textProc/training_data.txt"
    --     textFilePath 
    --     wordlst
    --     wordToReadInFile
    -- putStrLn $ "Data preprocessing duration: " ++ show duration ++ "s"
    -- ----------------- TRAINING -----------------

    (trainingData, loadingDuration) <- chronometer $ loadTrainingData "data/textProc/training_data.txt" wordToOneHot 100000
    putStrLn $ show (length trainingData) ++ " training datas were grabbed in " ++ show loadingDuration ++ "s"
    -- initModel <- loadParams hypParams modelPath
    -- initModel <- sample hypParams
    -- model <- trainModel initModel trainingData epochNb 40000
    -- -- ----------------- TESTING -----------------
    -- model <- loadParams hypParams "data/textProc/1000w_100000trainset/sample_embedding.params"
    -- let word2vec = zip wordlst $ (asValue (toDependent $ weight $ w2 model) :: [[Float]])
    -- let word2vecDict = M.fromList word2vec
    -- let wordVec1 = asTensor $ fromJust $ M.lookup (TL.encodeUtf8 $ TL.pack "phone") word2vecDict
    -- let wordVec2 = asTensor $ fromJust $ M.lookup (TL.encodeUtf8 $ TL.pack "small") word2vecDict
    -- let wordVec3 = asTensor $ fromJust $ M.lookup (TL.encodeUtf8 $ TL.pack "big") word2vecDict
    -- let subVec = Torch.Functional.add wordVec3 (Torch.Functional.sub wordVec1 wordVec2)
    -- let res = asValue subVec :: [Float]
    -- print $ mostSimilar res word2vec
    -- print $ mostSimilarWord "software" word2vec word2vecDict
    -- print $ mostSimilarWord "android" word2vec word2vecDict
    -- print $ mostSimilarWord "week" word2vec word2vecDict
    -- print $ mostSimilarWord "man" word2vec word2vecDict
    -- let lay = weight $ w1 model
    -- print lay 
    -- let !epochLoss = loss model (wordToOneHot (TL.encodeUtf8 $ TL.pack "phone") , wordToOneHot (TL.encodeUtf8 $ TL.pack "small"))
    -- print epochLoss
    -- (t, o) <- runStep model GD (asTensor epochLoss) 0.1
    -- print lay 
    return ()

exportTrainingData :: FilePath -> FilePath -> [B.ByteString] -> Int  -> IO [(B.ByteString, B.ByteString)]
exportTrainingData filePath trainingText wordlst wordToReadInFile  = do
    putStrLn "grabbing training data..."
    
    rng <- newStdGen
    trainingText <- B.readFile textFilePath

    let trainingTextLines = take wordToReadInFile $ preprocess trainingText
        filteredTrainingTextLines = map (filter (\x -> x `elem` wordlst)) trainingTextLines    
    print $ length filteredTrainingTextLines
    
    let total = length filteredTrainingTextLines
    
    trainingDataPackList <- mapM (\(i, line) -> do
                       let result = packOfFollowingWords line
                       putStrLn $ show (i + 1)
                       return result
                   ) (zip [0..] filteredTrainingTextLines)
    -- putStrLn $ (show $ length trainingDataPackList) ++ " training data created"
    -- let trainingDataPack = concat $ map packOfFollowingWords filteredTrainingTextLines
    spacedTrainingData <- mapM (\(i, line) -> do
                       let result = map (\(bs1,bs2) -> bs1 `B.append` (B.pack $ encode " ") `B.append` bs2) line
                       when (i `mod` 100 == 0) $
                           putStrLn $ "spacing line nb : " ++ (show (i + 1)) ++ "/" ++ (show $ length trainingDataPackList) ++ " - length : " ++ (show $ length result)
                       return result
                   ) (zip [0..] trainingDataPackList)
    -- print $ "spaced " ++ (show $ length spacedTrainingData)
    let trainingDataPack = concat spacedTrainingData
    -- let shuffledTrainingData = shuffle' trainingDataPack (length trainingDataPack) rng
    print $ "shuffled " ++ (show $ length trainingDataPack)
    B.writeFile filePath (B.intercalate (B.pack $ encode "\n") trainingDataPack)
    putStrLn "training data saved"
    return $ concat trainingDataPackList

loadTrainingData :: FilePath -> (B.ByteString -> Tensor) -> Int -> IO [(Tensor, Tensor)]
loadTrainingData trainingDataPath wordToOneHot maxTrainingDataSize = do
    trainingDataWords <- B.readFile trainingDataPath
    rng <- newStdGen 
    let fileTextLines = take maxTrainingDataSize $ B.split (head $ encode "\n") trainingDataWords
    trainingData <- mapM (\(i,x) -> do
        let words = B.split (head $ encode " ") x
            word1 = head words
            word2 = head $ tail words
            !word1OneHot = wordToOneHot word1
            !word2OneHot = wordToOneHot word2
        when (i `mod` 100 == 0) $
            putStrLn $ "loading training data " ++ (show $ 100 * (fromIntegral i) / (fromIntegral $ length fileTextLines)) ++ "%"
        return (word1OneHot, word2OneHot)
        ) (zip [0..] fileTextLines)
    return trainingData


-- | Find the most similar words to a given word
mostSimilarWord ::
    String
    -> [(B.ByteString, [Float])]
    -> M.Map B.ByteString [Float]
    -> [(B.ByteString, Float)]
mostSimilarWord word word2vec word2vecDict =
    case M.lookup lazyWord word2vecDict of
        Nothing -> []
        Just wordVec -> mostSimilar wordVec word2vec
    where
        lazyWord = TL.encodeUtf8 $ TL.pack word

-- | Find the most similar vectors to a given vector
mostSimilar ::
    [Float]
    -> [(B.ByteString, [Float])]
    -> [(B.ByteString, Float)]
mostSimilar wordVec word2vec = 
        take 10 $ 
        reverse $
        sortBySnd $ map (\(w, vec) -> (w, similarityCosine wordVec vec)) word2vec

trainModel :: EmbeddingParams -> [(Tensor, Tensor)] -> Int -> Int -> IO EmbeddingParams
trainModel model trainingData epochNb batchSize = do
    putStrLn "Training model..."
    let optimizer = mkAdam 0 0.9 0.999 (flattenParameters model)
    (trainedModel, _, losses, _) <-
        foldLoop (model, optimizer, [], trainingData) epochNb $ \(model, opt, losses, datas) i -> do
            let epochLoss = sum (map (loss model) (take batchSize datas))
            let lossValue = asValue epochLoss :: Float
            putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue
            (trainedModel, nOpt) <- runStep model opt epochLoss 0.1
            pure (trainedModel, nOpt, losses ++ [lossValue], drop batchSize datas ++ take batchSize datas) -- pure : transform return type to IO because foldLoop need it
    saveParams trainedModel modelPath
    return trainedModel

wordToOneHotLookupFactory :: Int -> Tensor
wordToOneHotLookupFactory wordNum = stack (Dim 0) $ map (\i -> oneAtPos i wordNum) [0..(wordNum - 1)]

-- 100000 in [125, 123] sec
wordToOneHotFactory ::  (B.ByteString -> Int) -> Tensor -> (B.ByteString -> Tensor)
wordToOneHotFactory wordToIndex wordToOneHotLookup word = embedding' wordToOneHotLookup (asTensor [[wordToIndex word :: Int]])

-- 100000 in [128, 129] sec
wordToOneHot' :: [B.ByteString] -> B.ByteString -> Tensor
wordToOneHot' wordlst word = oneAtPos index (length wordlst)
    where
        wordToIndex = wordToIndexFactory wordlst
        index       = wordToIndex word


preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (map (\s -> toByteString $ map toLower $ toString s)) words
    where
        toByteString = TL.encodeUtf8 . TL.pack
        toString     = TL.unpack . TL.decodeUtf8
        replacedTexts = B.pack $ map replaceUnnecessaryChar (B.unpack texts)
        textLines     = B.split (head $ encode "\n") replacedTexts
        wordsLst      = map (B.split (head $ encode " ")) textLines
        words         = map (filter (not . B.null)) $ wordsLst

wordToIndexFactory ::
    [B.ByteString] -- wordlist
    -> (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd =
    M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0 ..]))

packOfFollowingWords :: [B.ByteString] -> [(B.ByteString, B.ByteString)] -- check if length of wordlst is greater than n
packOfFollowingWords [] = []
packOfFollowingWords [x] = []
packOfFollowingWords (x:xs) =
    if length xs > 2
        then (p2, p1) : (p3, p1) : (p1, p2) : (p1, p3) : packOfFollowingWords xs
        else (p2, p1) : (p1, p2) : []
    where
        p1 = x
        p2 = head xs
        p3 = head $ tail xs

loadWordLst :: FilePath -> IO [B.ByteString]
loadWordLst wordLstPath = do
    texts <- B.readFile wordLstPath
    let wordlst = B.split (head $ encode "\n") texts
    return wordlst

extractWordLst :: FilePath -> Int -> Int -> IO [B.ByteString]
extractWordLst textFilePath linesToReadInFile wordNbToGrab = do -- load text file
    texts <- B.readFile textFilePath
    putStrLn "loaded text file"
            
    -- create word lst (unique)
    let wordLines = take linesToReadInFile $ preprocess texts
        allWords = concat wordLines

        wordCountMap = countWords allWords
        wordFrequentSorted = sortByFrequency wordCountMap
        wordFrequentTop = take wordNbToGrab wordFrequentSorted
        wordlst = map fst wordFrequentTop
    
    putStrLn "created word list"
        -- save word list
    B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
    putStrLn "saved word list"
    return wordlst


-- Function to count occurrences of words in a list
countWords :: [B.ByteString] -> M.Map B.ByteString Int
countWords = foldl' (\acc word -> M.insertWith (+) word 1 acc) M.empty

-- Function to sort by the frequency of words
sortByFrequency :: M.Map B.ByteString Int -> [(B.ByteString, Int)]
sortByFrequency = sortBy (comparing (negate . snd)) . M.toList
