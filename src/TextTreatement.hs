module TextTreatement where

import           Data.Word
import           Codec.Binary.UTF8.String     (encode)
import           Text.CSV
import           System.Directory

-- Function to parse CSV file or throw an error
parseCSVOrError :: String -> IO CSV
parseCSVOrError filePath = do
    currentDir <- getCurrentDirectory
    csv <- parseCSVFromFile (currentDir ++ "/" ++ filePath)
    case csv of
        Right csvData -> return csvData
        Left err -> error $ "Error parsing CSV: " ++ show err



isAlphaNum :: Char -> Bool
isAlphaNum c = c `elem` ['0'..'9'] || c `elem` ['a'..'z'] || c `elem` ['A'..'Z']

isSpace :: Char -> Bool
isSpace = (`elem` [' ', '\t', '\n', '\r', '\f', '\v'])

isEmoji :: Char -> Bool
isEmoji c = c >= '\x1F600' && c <= '\x1F64F'

isUnnecessaryChar :: Word8 -> Bool
isUnnecessaryChar w = let c = toEnum (fromEnum w) :: Char
                        in not (isAlphaNum c) && not (isSpace c) && not (isEmoji c)


replaceUnnecessaryChar :: Word8 -> Word8
replaceUnnecessaryChar w = if isUnnecessaryChar w then (head $ encode " ") else w

