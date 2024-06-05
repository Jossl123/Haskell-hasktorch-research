{-# LANGUAGE DeriveAnyClass        #-}
{-# LANGUAGE DeriveGeneric         #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings     #-}
{-# LANGUAGE RecordWildCards       #-}
{-# LANGUAGE BangPatterns #-}

module Main
    ( main
    ) where

import           Codec.Binary.UTF8.String (encode)
import           Data.Aeson               (FromJSON (..), ToJSON (..),
                                           eitherDecode)
import qualified Data.ByteString.Internal as B (c2w)
import qualified Data.ByteString.Lazy     as B
import           GHC.Generics
import           Torch.Autograd           (makeIndependent, toDependent)
import           Torch.NN                 (Parameter, Parameterized (..),
                                           Randomizable (..))
import           Torch.Serialize          (loadParams)
import           Torch.TensorFactories    (randnIO', zeros', ones')    
import           Torch.Device
import           Torch.Functional         (Dim (..), mseLoss, sumAll, embedding', stack, mul )
import           Torch.Tensor             (Tensor, asTensor, asValue, reshape, shape)
import           Torch.Train              (saveParams)
import           Torch.Layer.Linear       (LinearHypParams (..), LinearParams (..), linearLayer)
import           Word2Vec                 (EmbeddingParams (..), EmbeddingHypParams (..), wordToOneHotFactory, wordToOneHotLookupFactory, wordToIndexFactory, preprocess) 


import           Torch.Layer.NonLinear    (ActName (..))
import           Torch.Optim                   (Adam (..), GD (..), foldLoop,
                                               mkAdam, runStep)
            
import           Torch.Layer.RNN           (RnnHypParams(..), RnnParams(..), rnnLayers)

import           Control.Monad        (when)
import Data.Text.Lazy             as TL  (Text, pack)
import Data.Text.Lazy.Encoding    as TL  (encodeUtf8)

import Debug.Trace  (trace)

-- amazon review data
data Image =
    Image
        { small_image_url  :: String
        , medium_image_url :: String
        , large_image_url  :: String
        }
    deriving (Show, Generic)

instance FromJSON Image

instance ToJSON Image

data AmazonReview =
    AmazonReview
        { rating            :: Float
        , title             :: String
        , text              :: String
        , images            :: [Image]
        , asin              :: String
        , parent_asin       :: String
        , user_id           :: String
        , timestamp         :: Int
        , verified_purchase :: Bool
        , helpful_vote      :: Int
        }
    deriving (Show, Generic)

instance FromJSON AmazonReview

instance ToJSON AmazonReview

-- model
data ModelSpec =
    ModelSpec
        { wordNum :: Int, -- the number of words
          wordDim :: Int, -- the dimention of word embeddings
          hiddenDim :: Int,
          numLayers :: Int    
        }
    deriving (Show, Eq, Generic)

data Model =
    Model
        { emb :: EmbeddingParams,
          rnn :: RnnParams
        }
    deriving (Show, Generic, Parameterized)

instance Randomizable ModelSpec Model where
    sample ModelSpec {..} = do
        emb_sampled <- sample EmbeddingHypParams {dev = Device CPU 0, wordDim = wordDim, wordNum = wordNum}
        rnn_sampled <- sample $ RnnHypParams {dev = Device CPU 0, bidirectional = False, inputSize = wordDim, hiddenSize = hiddenDim, numLayers = numLayers, hasBias = True}
        return Model {emb = emb_sampled, rnn = rnn_sampled}
            

-- randomize and initialize embedding with loaded params
initialize :: ModelSpec -> FilePath -> IO Model
initialize modelSpec embPath = do
    randomizedModel <- sample modelSpec
    loadedEmb <- loadParams (emb randomizedModel) embPath
    return Model {emb = loadedEmb, rnn = rnn randomizedModel}



asFloat :: Tensor -> Float
asFloat t = asValue t :: Float

-- your amazon review json
amazonReviewPath :: FilePath
amazonReviewPath = "data/amazon/train.jsonl"

outputPath :: FilePath
outputPath = "data/amazon/review-texts.txt"

embeddingPath = "data/textProc/1000w_100000trainset/sample_embedding.params"

wordLstPath = "data/textProc/1000w_100000trainset/sample_wordlst.txt"

word2vecFactory :: Model -> (B.ByteString -> Int) -> (B.ByteString -> Tensor)
word2vecFactory model wordToIndex word  = reshape [last $ shape embedded] embedded
    where embedded = embedding' (toDependent $ weight $ w2 $ emb model) $ asTensor [[wordToIndex word :: Int]]

decodeToAmazonReview :: B.ByteString -> Either String [AmazonReview]
decodeToAmazonReview jsonl =
    let jsonList = B.split (B.c2w '\n') jsonl
    in sequenceA $ map eitherDecode jsonList

amazonReviewToTrainingData :: [AmazonReview] -> [B.ByteString] -> (B.ByteString -> Tensor) -> [(Tensor, Tensor)]
amazonReviewToTrainingData reviews wordLst word2vec = map (\review -> (input review, output review)) reviews
    where
        seqLength = 15
        input review = stack (Dim 0) $ addPadding $ sentenceEmbedded review
        sentenceEmbedded review = map word2vec $ processedSentence review
        output review = asTensor $ replicate seqLength [rating review]
        addPadding sentence = sentence ++ replicate (seqLength - (length sentence)) (asTensor (replicate 16 (0.0 :: Float)))
        processedSentence review = take seqLength $ filterWordNotInWordLst $ concat $ preprocess ( TL.encodeUtf8 . TL.pack $ text review) 
        filterWordNotInWordLst sentence = filter (\x -> x `elem` wordLst) sentence

main :: IO ()
main = do
    jsonl <- B.readFile amazonReviewPath
    -- load word list (It's important to use the same list as whan creating embeddings)
    wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
    let amazonReviews = decodeToAmazonReview jsonl
        wordToIndex = wordToIndexFactory wordLst
        wordToOneHotLookup = wordToOneHotLookupFactory (length wordLst)
        wordToOneHot = wordToOneHotFactory wordToIndex wordToOneHotLookup
        reviews =
            case amazonReviews of
            Left err      -> []
            Right reviews -> reviews
    
    -- load params (set　wordDim　and wordNum same as session5)
    let modelSpec = ModelSpec {wordDim = 16, wordNum = 1000, hiddenDim = 1, numLayers = 2}
    initModel <- initialize modelSpec embeddingPath

    let trainingData = amazonReviewToTrainingData (take 2000 reviews) wordLst (word2vecFactory initModel wordToIndex)
    -- print reviews
    -- print $ head trainingData
    trainedModel <- trainModel (rnn initModel) trainingData 1000 10000

    -- load model 
    let modelPath = "app/amazon/models/amazon_20_18%_524loss.model"
    model <- loadParams (rnn initModel) modelPath

    -- test model
    let input = (fst $ head $ tail $ tail trainingData)
        input2 = (fst $ head $ tail $ tail $ tail $ tail $tail trainingData)
    print input
    print input2
    print $  forward model input 
    print $  forward model input2 
    print $ snd $ head $ tail $ tail trainingData
    print $ snd $ head $ tail $ tail $ tail $tail $ tail trainingData

    let testData = take 10 trainingData
        outputs = map (\x -> last $ asValue (forward model $ fst x) :: [Float]) testData
        targets = map (\x -> last $ asValue (snd x) :: [Float]) testData

    print $ zip outputs targets

    -- print trainedModel 
    return ()


loss :: RnnParams -> (Tensor, Tensor) -> Tensor
loss model (input, target) = sumAll $ mseLoss ((forward model input) `mul` (asTensor ([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1] :: [Float]))) (target `mul` (asTensor ([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1] :: [Float])))

forward :: RnnParams -> Tensor -> Tensor
forward model input = fst $ rnnLayers model Relu (Just 0.8) (ones' [2, 1]) input

trainModel :: RnnParams -> [(Tensor, Tensor)] -> Int -> Int -> IO RnnParams
trainModel model trainingData epochNb batchSize = do
    putStrLn "Training model..."
    print $ length trainingData
    let optimizer = mkAdam 0 0.9 0.999 (flattenParameters model)
        validationSetLength = length trainingData `div` 10
        (validationSet, dataSet) = splitAt validationSetLength trainingData
    (trainedModel, _, losses, _) <-
        foldLoop (model, optimizer, [], dataSet) epochNb $ \(model, opt, losses, datas) i -> do
            let epochLoss = sum (map (loss model) (take batchSize datas))
            let lossValue = asValue epochLoss :: Float
            putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue
            (trainedModel, nOpt) <- runStep model opt epochLoss 0.01
            when (i `mod` 10 == 0) $ do
                putStrLn "Saving..."

                let res = sum $ map (\(input, target) -> if (round $ asFloat $ forward model input) == (round $ asFloat target) then 1 else 0) validationSet
                    accuracy = res / fromIntegral validationSetLength
                saveParams trainedModel ("app/amazon/models/amazon_" ++ show (i + 0) ++ "_" ++ show (round (100 * accuracy)) ++ "%_" ++ show (round lossValue) ++ "loss.model" )
                -- drawLearningCurve "app/amazon/models/graph-amazon_.png" "Learning Curve" [("learningRate", losses)]
                putStrLn "Saved..."
                -- let currencyPrices = reverse $ map price currencyDatas
                    -- guesses = reverse $ map (\(input, _) -> asValue $ forward model input) trainingDatas
                -- drawLearningCurve "app/trading/models/graph-currency.png" "Currency Price" [("price", currencyPrices),("price", (replicate (4956-(length trainingDatas)) 0) ++ guesses)]
                
            pure (trainedModel, nOpt, losses ++ [lossValue], drop batchSize datas ++ take batchSize datas) -- pure : transform return type to IO because foldLoop need it
    -- saveParams trainedModel modelPath
    return trainedModel