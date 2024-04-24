module Utils (parseCSVOrError) where

import Text.CSV
import System.Directory (getCurrentDirectory)


-- Function to parse CSV file or throw an error
parseCSVOrError :: String -> IO CSV
parseCSVOrError filePath = do
    currentDir <- getCurrentDirectory
    csv <- parseCSVFromFile (currentDir ++ "/" ++ filePath)
    case csv of
        Right csvData -> return csvData
        Left err -> error $ "Error parsing CSV: " ++ show err