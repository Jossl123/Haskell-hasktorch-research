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
import           Torch.TensorFactories    (randnIO', zeros')
import           Torch.Device
import           Torch.Functional         (mseLoss, sumAll)
import           Torch.Tensor             (Tensor, asTensor, asValue)
import           Torch.Train              (saveParams)

import           Word2Vec                 (EmbeddingParams (..), EmbeddingHypParams (..), wordToOneHotFactory, wordToOneHotLookupFactory, wordToIndexFactory, preprocess) 

import           Torch.Layer.NonLinear    (ActName (..))
import           Torch.Optim                   (Adam (..), GD (..), foldLoop,
                                               mkAdam, runStep)
import           Torch.Layer.RNN           (RnnHypParams(..), RnnParams(..), rnnLayers)


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
          wordDim :: Int -- the dimention of word embeddings
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
        rnn_sampled <- sample $ RnnHypParams {dev = Device CPU 0, bidirectional = False, inputSize = wordDim, hiddenSize = 1, numLayers = 2, hasBias = True}
        return Model {emb = emb_sampled, rnn = rnn_sampled}
            

-- randomize and initialize embedding with loaded params
initialize :: ModelSpec -> FilePath -> IO Model
initialize modelSpec embPath = do
    randomizedModel <- sample modelSpec
    loadedEmb <- loadParams (emb randomizedModel) embPath
    rnn <- sample $ RnnHypParams {dev = Device CPU 0, bidirectional = False, inputSize = 16, hiddenSize = 1, numLayers = 2, hasBias = True}
    return Model {emb = loadedEmb, rnn = rnn }

-- your amazon review json
amazonReviewPath :: FilePath
amazonReviewPath = "data/amazon/train.jsonl"

outputPath :: FilePath
outputPath = "data/amazon/review-texts.txt"

embeddingPath = "data/textProc/1000w_100000trainset/sample_embedding.params"

wordLstPath = "data/textProc/1000w_100000trainset/sample_wordlst.txt"

decodeToAmazonReview :: B.ByteString -> Either String [AmazonReview]
decodeToAmazonReview jsonl =
    let jsonList = B.split (B.c2w '\n') jsonl
    in sequenceA $ map eitherDecode jsonList

amazonReviewToTrainingData :: [AmazonReview] -> (B.ByteString -> Tensor) -> [(Tensor, Tensor)]
amazonReviewToTrainingData reviews wordToOneHot = map (\review -> (input review, output review)) reviews
    where
        input review = asTensor [[rating review],[rating review]]
        output review = asTensor [[rating review]]

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
    let modelSpec = ModelSpec {wordDim = 16, wordNum = 1000}
    initModel <- initialize modelSpec embeddingPath

    let trainingData = amazonReviewToTrainingData reviews wordToOneHot
    -- print reviews
    trainedModel <- trainModel (rnn initModel) trainingData 10 10

    -- print trainedModel
    return ()


loss :: RnnParams -> (Tensor, Tensor) -> Tensor
loss model (input, target) = sumAll $ mseLoss output target
    where
        (output, _) = rnnLayers model Tanh (Just 0.8) input (zeros' [1, 16])

trainModel :: RnnParams -> [(Tensor, Tensor)] -> Int -> Int -> IO RnnParams
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
    -- saveParams trainedModel modelPath
    return trainedModel