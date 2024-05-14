{-# LANGUAGE OverloadedStrings #-}

-- import Control.Monad (forever)
-- import qualified Data.ByteString as BS
-- import qualified Data.ByteString.Char8 as C8
import Opencv

-- captureImage :: FilePath -> IO ()
-- captureImage dev = do
--     withDevice dev $ \fd -> do
--         caps <- queryCapability fd
--         putStrLn $ "Driver: " ++ show (capabilityDriver caps)
--         putStrLn $ "Card: " ++ show (capabilityCard caps)
--         putStrLn $ "Bus info: " ++ show (capabilityBusInfo caps)
--         fmt <- getFormat fd
--         putStrLn $ "Format: " ++ show fmt
--         case fmt of
--             ImageFormatRGB24 width height -> do
--                 putStrLn $ "Image size: " ++ show (width, height)
--                 forever $ do
--                     img <- readImageRGB24 fd width height
--                     -- Process or save the image as needed
--                     putStrLn "Image captured"
--             _ -> putStrLn "Unsupported image format"


main :: IO ()
main = do
    putStrLn "Starting camera capture..."
    -- captureImage "/dev/video0"
