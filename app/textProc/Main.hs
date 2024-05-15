-- PLEASE DOWNLOAD THE REQUIRED DATAS IF YOU WANT TO RUN THIS PROGRAM 
-- https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download

module Main where

import Functional           (sortBySnd)
import Torch.Model.Utils    (accuracy, f1, precision, recall, macroAvg, weightedAvg)
import Torch.Tensor.Util    (indexOfMax, oneHot')
import qualified Data.Text as T

import Graphics.Matplotlib
import System.Random
import Data.List.Split
import Data.List (elemIndex)
import Data.Maybe (fromJust)
import Torch.Optim          (foldLoop)
import ReadImage            (imageToRGBList)

import Control.Monad        (when)
import Data.List            (sortBy, maximumBy)

import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.TensorFactories (zeros')
import Torch.Functional     (mseLoss, Dim(..), exp, sumAll, div)
import Torch.NN             (sample,flattenParameters)
import Torch.Optim          (GD(..), Adam(..), mkAdam, runStep, foldLoop)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Autograd       (toDependent)
import Torch.Layer.MLP      (MLPHypParams(..), MLPParams(..), ActName(..), mlpLayer)
import Torch.Layer.Linear   (LinearParams(..))
import ML.Exp.Chart         (drawLearningCurve,drawConfusionMatrix ) --nlp-tools
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

loss :: MLPParams -> (Tensor, Tensor) -> Tensor
loss model (input, output) = let y = mlpLayer model input
                             in mseLoss y output

main :: IO ()
main = do
    let device = Device CPU 0
        epochNb = 30
        wordDim = 32
        wordNum = 8198
        wordToReadInFile = 100000
        hypParams = MLPHypParams device wordNum [(wordDim, Id),(wordNum, Softmax)] 

    -- extractWordLst textFilePath wordToReadInFile wordNum
    wordlst <- loadWordLst wordLstPath
    trainingText <- B.readFile textFilePath

    -- initModel <- sample hypParams

    -- let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initModel)
    --     trainingTextWords = take wordToReadInFile $ preprocess trainingText
    -- putStrLn "grabbing training data..."
    -- trainingData <- getTrainingData wordlst (packOfFollowingWords trainingTextWords 1)
    -- putStrLn "start training"
    -- (trainedModel, _, losses) <- foldLoop (initModel, optimizer, []) epochNb $ \(model, opt, losses) i -> do 
    --     let epochLoss = sum (map (loss model) trainingData)
    --     let lossValue = asValue epochLoss :: Float
    --     putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue 
    --     (trainedModel, nOpt) <- runStep model opt epochLoss 0.1
    --     pure (trainedModel, nOpt, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it 
    -- saveParams trainedModel modelPath


    model <- loadParams hypParams modelPath

    let input = wordToOneHot (TL.encodeUtf8 $ TL.pack "device") wordlst
    -- print input
    -- print $ mlpLayer initModel input
    let output = mlpLayer model input
    let outputWords = zip wordlst $ (asValue output :: [Float])


    let word2vec = zip wordlst $ (asValue (toDependent $ weight $ fst $ head $ tail $ layers model) :: [[Float]])
    let word2vecDict = M.fromList word2vec
    let me = asTensor $ fromJust $ M.lookup (TL.encodeUtf8 $ TL.pack "me") word2vecDict 
    let this =asTensor $ fromJust $ M.lookup (TL.encodeUtf8 $ TL.pack "android") word2vecDict 
    let you =asTensor $ fromJust $ M.lookup (TL.encodeUtf8 $ TL.pack "you") word2vecDict 
    
    let res = asValue (this) :: [Float]

    let mostSim = take 10 $ reverse $ sortBySnd $ map (\(word, vec) -> (word, similarityCosine res vec)) word2vec
    print mostSim
    print $ take 10 $ reverse $ sortBySnd outputWords

    return ()

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
        [".", "!", "<", ">", "/", "\"", "(", ")", ":", ";", ",", "?", "@", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

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
packOfFollowingWords (x:xs) n = if length xs > n then (take n $ tail xs , head xs) : packOfFollowingWords xs n else []

-- Convert data into training data
getTrainingData :: [B.ByteString] -> [([B.ByteString], B.ByteString)] -> IO [(Tensor, Tensor)]
getTrainingData wordlst dataPack = do 
    let input x = asTensor $ concat $ map (\word -> asValue (oneAtPos word (length wordlst)) :: [Float]) (idxLst x)
        idxLst x = catMaybes $ map (`elemIndex` wordlst) x
        output y = asTensor $ concat $ map (\word -> asValue (oneAtPos word (length wordlst)) :: [Float]) (catMaybes [elemIndex y wordlst])
        res = map (\(x, y) -> (input x, output y)) dataPack
        filteredRes = filter (\(x, y) -> (length (asValue x :: [Float]) > 0) && (length (asValue y :: [Float]) > 0)) res
    return $ take 10000 $ filteredRes

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
