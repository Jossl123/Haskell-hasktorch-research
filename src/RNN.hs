
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module RNN (
    RnnParams(..),
    Rnn(..),
    singleLayerRnnForward,
    SingleLayerRnnParams(..),
    SingleLayerRnn(..),
    rnnForward
) where

import           GHC.Generics
import           Control.Monad                (forM)
import           Torch.NN
import           Torch.Device
import           Torch.Layer.Linear           (LinearHypParams (..), LinearParams (..), linearLayer)
import           Torch.Functional             (Dim (..), softmax, stack)
import           Torch.Tensor                 (Tensor (..) )
import           Torch.Layer.NonLinear        (ActName(..), decodeAct)
import           Torch.Tensor.Util            (unstack) 

data SingleLayerRnnParams = SingleLayerRnnParams {
    sdev :: Device,
    si_dim :: Int,
    sh_dim :: Int,
    so_dim :: Int,
    sactivation_name :: ActName
} deriving (Show, Generic)

data SingleLayerRnn = SingleLayerRnn {
    i_linear :: LinearParams,
    h_linear :: LinearParams,
    o_linear :: LinearParams,
    activation :: Tensor -> Tensor
} deriving (Generic)

instance Parameterized SingleLayerRnn
    

instance Show SingleLayerRnn where
    show SingleLayerRnn {..} = "SingleLayerRnn show TODO"

instance Randomizable SingleLayerRnnParams SingleLayerRnn
    where
    sample SingleLayerRnnParams {..} = do
        i <- sample $ LinearHypParams sdev True si_dim sh_dim
        h <- sample $ LinearHypParams sdev True sh_dim sh_dim
        o <- sample $ LinearHypParams sdev True sh_dim so_dim
        return $ SingleLayerRnn i h o $ decodeAct sactivation_name

data RnnParams = RnnParams {
    dev :: Device,
    h_dim :: Int,
    i_dim :: Int,
    o_dim :: Int,
    num_layers :: Int,
    activation_name :: ActName
} deriving (Show, Generic)

data Rnn = Rnn {
    layers :: [SingleLayerRnn]
} deriving (Show, Generic)

instance Parameterized Rnn 


instance Randomizable RnnParams Rnn
    where
    sample RnnParams {..} = do
        let layer_param = SingleLayerRnnParams dev i_dim h_dim o_dim activation_name
        layers_sampled <- forM [1..num_layers] (\_ -> do sample layer_param)
        return $ Rnn layers_sampled

singleLayerRnnForward :: SingleLayerRnn -> Tensor -> Tensor -> (Tensor, Tensor)
singleLayerRnnForward SingleLayerRnn {..} input hidden = (output, hidden')
    where
        hidden' = activation $ linearLayer h_linear hidden + linearLayer i_linear input
        output = linearLayer o_linear hidden'

singleLayerRnnForwardNoHidden :: SingleLayerRnn -> Tensor -> (Tensor, Tensor)
singleLayerRnnForwardNoHidden SingleLayerRnn {..} input = (output, hidden')
    where
        hidden' = activation $ linearLayer i_linear input
        output = linearLayer o_linear hidden'

rnnForward' :: Rnn -> [Tensor] -> Tensor -> ([Tensor], [Tensor])
rnnForward' Rnn {..} inputs hidden = (outputs, tail hiddens)
    where
        (outputs, hiddens) = foldl step ([], [hidden]) $ zip layers inputs
        step (os, hs) (layer, input) = let (o, h) = singleLayerRnnForward layer input $ last hs in (os ++ [o], hs ++ [h])

rnnForward :: Rnn -> Tensor -> (Tensor, Tensor)
rnnForward Rnn {..} input_tensor = (stack (Dim 0) outputs, stack (Dim 0) hiddens)
    where
        inputs = unstack input_tensor
        (first_output, first_hidden) = singleLayerRnnForwardNoHidden (head layers) $ head inputs
        (outputs, hiddens) = foldl step ([first_output], [first_hidden]) $ zip (tail layers) (tail inputs)
        step (os, hs) (layer, input) = let (o, h) = singleLayerRnnForward layer input $ last hs in (os ++ [o], hs ++ [h])