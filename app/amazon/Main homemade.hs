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
import           Torch.TensorFactories    (randnIO')
import           Torch.Device
import Torch.Functional (mseLoss)
import           Torch.Tensor             (Tensor, asTensor, asValue)

import           RNN                     (Rnn(..), RnnParams (..), rnnForward, SingleLayerRnn(..))
import           Word2Vec                (EmbeddingParams (..), EmbeddingHypParams (..))
import           Torch.Layer.NonLinear   (ActName (..))
import           Torch.Optim                  (Adam (..), GD (..), foldLoop,
                                               mkAdam, runStep, grad')
import           Torch.Layer.Linear       (LinearParams (..), LinearHypParams (..), linearLayer)
import           Torch.Tensor.Util            (unstack) 

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
        { wordNum :: Int 
    -- the number of words
        , wordDim :: Int -- the dimention of word embeddings
        }
    deriving (Show, Eq, Generic)

data Model =
    Model
        { emb :: EmbeddingParams
    -- TODO: add RNN
        --   rnn :: RNN
        }
    deriving (Show, Generic, Parameterized)

-- data RNN = RNN {    
--         firstRnnParams :: SingleRnnParams, -- ^ a model for the first RNN layer
--         restRnnParams :: [SingleRnnParams] -- ^ models for the rest of RNN layers
--     } deriving (Show, Generic)
-- instance Parameterized RnnParams

instance Randomizable ModelSpec Model where
    sample ModelSpec {..} =
        Model <$> sample EmbeddingHypParams {dev = Device CPU 0, wordDim = wordDim, wordNum = wordNum}
            -- TODO: add RNN initilization
            -- <*> sample ...

-- randomize and initialize embedding with loaded params
initialize :: ModelSpec -> FilePath -> IO Model
initialize modelSpec embPath = do
    randomizedModel <- sample modelSpec
    loadedEmb <- loadParams (emb randomizedModel) embPath
    return Model {emb = loadedEmb} --, rnn = rnn randomizedModel }


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

main :: IO ()
main = do
    -- jsonl <- B.readFile amazonReviewPath
    -- let amazonReviews = decodeToAmazonReview jsonl
    -- let reviews =
    --         case amazonReviews of
    --         Left err      -> []
    --         Right reviews -> reviews
    
    -- -- load word list (It's important to use the same list as whan creating embeddings)
    -- wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
    -- -- load params (set　wordDim　and wordNum same as session5)
    -- let modelSpec = ModelSpec {wordDim = 16, wordNum = 1000}
    -- initModel <- initialize modelSpec embeddingPath
    -- print initModel
    initModel <- sample $ RnnParams {dev = Device CPU 0, h_dim = 3, i_dim = 1, o_dim = 1, num_layers = 3, activation_name = Tanh}
    input <- randnIO' [3, 1] 
    let opt = mkAdam 0 0.9 0.999 (flattenParameters $ head $ layers initModel)
    let lay = weight $ i_linear $ head $ layers initModel
    print lay
    let (outputs, hiddens) = rnnForward initModel $ unstack input
    let (!outputs2, !hiddens2) = rnnForward initModel $ unstack input
    -- let !testee = map (\single -> (toDependent $ weight $ i_linear single, toDependent $ weight $ h_linear single, toDependent $ weight $ o_linear single)) $ layers initModel
    print outputs
    print outputs2
    print hiddens
    print hiddens2
    print lay 
    let epochLoss = mseLoss input outputs
    let epochLoss2 = mseLoss input outputs
    print epochLoss
    print epochLoss2
    (model, opt) <- runStep initModel GD epochLoss 0.1
    -- (trainedModel, nOpt) <- runStep (head $ layers initModel) opt (asTensor [200::Float]) 0.1
    -- print trainedModel
    return ()



-- trainModel :: EmbeddingParams -> [(Tensor, Tensor)] -> Int -> Int -> IO EmbeddingParams
-- trainModel model trainingData epochNb batchSize = do
--     putStrLn "Training model..."
--     let optimizer = mkAdam 0 0.9 0.999 (flattenParameters model)
--     (trainedModel, _, losses, _) <-
--         foldLoop (model, optimizer, [], trainingData) epochNb $ \(model, opt, losses, datas) i -> do
--             let epochLoss = sum (map (loss model) (take batchSize datas))
--             let lossValue = asValue epochLoss :: Float
--             putStrLn $ "Loss epoch " ++ show i ++ " : " ++ show lossValue
--             (trainedModel, nOpt) <- runStep model opt epochLoss 0.1
--             pure (trainedModel, nOpt, losses ++ [lossValue], drop batchSize datas ++ take batchSize datas) -- pure : transform return type to IO because foldLoop need it
--     saveParams trainedModel modelPath
--     return trainedModel