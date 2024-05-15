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

import Torch.Optim          (foldLoop)
import ReadImage            (imageToRGBList)

import Control.Monad        (when)
import Data.List            (sortBy, maximumBy)

import Torch.Tensor         (asTensor, asValue, Tensor(..))
import Torch.Functional     (mseLoss, Dim(..), exp, sumAll, div)
import Torch.NN             (sample,flattenParameters)
import Torch.Optim          (GD(..), Adam(..), mkAdam, runStep, foldLoop)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP      (MLPHypParams(..), MLPParams(..), ActName(..), mlpLayer)
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
    wordlst <- loadWordLst wordLstPath
    let device = Device CPU 0
        epochNb = 20 
        wordDim = 5
        wordNum = 10
        hypParams = MLPHypParams device wordNum [(wordDim, Id),(wordNum, Softmax)] -- Id | Sigmoid | Tanh | Relu | Elu | Selu

    initModel <- sample hypParams
    putStrLn "start training"
    print $ wordlst
    print $ (packOfFollowingWords wordlst 1)
    let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initModel)
        trainingData = getTrainingData wordlst (packOfFollowingWords wordlst 1)
    print trainingData
    (trainedModel, _, losses) <- foldLoop (initModel, optimizer, []) epochNb $ \(model, opt, losses) i -> do 
        let epochLoss = sum (map (loss model) trainingData)
        let lossValue = asValue epochLoss :: Float
        putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue 
        (trainedModel, nOpt) <- runStep model opt epochLoss 0.002
        pure (trainedModel, nOpt, losses ++ [lossValue]) -- pure : transform return type to IO because foldLoop need it 

    return ()


-- your text data (try small data first)
textFilePath = "data/textProc/review-texts.txt"

modelPath = "data/textProc/sample_embedding.params"

wordLstPath = "data/textProc/sample_wordlst.txt"


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

count :: Eq a => a -> [a] -> Int
count x = length . filter (x ==)

packOfFollowingWords :: [B.ByteString] -> Int -> [([B.ByteString], B.ByteString)] -- check if length of wordlst is greater than n
packOfFollowingWords [] _ = []
packOfFollowingWords [x] _ = []
packOfFollowingWords (x:xs) n = if length xs > n then (take n $ tail xs , head xs) : packOfFollowingWords xs n else []

-- Convert data into training data
getTrainingData :: [B.ByteString] -> [([B.ByteString], B.ByteString)] -> [(Tensor, Tensor)]
getTrainingData wordlst dataPack = map (\(x, y) -> (input x, output y)) dataPack
    where input x = asTensor $ concat $ map (\word -> asValue (oneAtPos word (length wordlst)) :: [Float]) (idxLst x)
          idxLst x = catMaybes $ map (`elemIndex` wordlst) x
          output y = asTensor $ concat $ map (\word -> asValue (oneAtPos word (length wordlst)) :: [Float]) (catMaybes [elemIndex y wordlst])

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
        
    -- save word list
    B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
    putStrLn "saved word list"
