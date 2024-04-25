module ReadImage (imageToRGBList) where 

import Codec.Picture
import Data.Matrix 

extractToRGBList :: DynamicImage -> [Float]
extractToRGBList dynamicImage =
    case dynamicImage of
        ImageRGB8 image -> rgbListFromImage image
        _ -> error "Unsupported image format"

rgbListFromImage :: Image PixelRGB8 -> [Float]
rgbListFromImage image =
    let (width, height) = (imageWidth image, imageHeight image)
        pixelList = [[(fromIntegral r) / 255.0, (fromIntegral g) / 255.0,(fromIntegral b) / 255.0] | y <- [0..height-1], x <- [0..width-1], let (PixelRGB8 r g b ) = pixelAt image x y]
    in concat pixelList

-- Function to convert an image to a list of RGB values
imageToRGBList :: String -> IO (Either String [Float])
imageToRGBList fileName = do
    image <- readImage fileName
    case image of
        Left _ -> return $ Left "error while loading image"
        Right dynamicImage -> return $ Right $ extractToRGBList dynamicImage
