module Utils (chronometer) where

import           Data.Time.Clock

chronometer :: IO a -> IO (a, Double)
chronometer action = do
    start <- getCurrentTime
    result <- action
    end <- getCurrentTime
    let diff = realToFrac $ diffUTCTime end start
    return (result, diff)