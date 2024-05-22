## Weather 

inputs : 7 pasts days temperatures, variance, temperature difference between last 2 days, slopeTrend, mean


## Titanic (kaggle 78,229%)

inputs : 

model : 7 -> (30, Relu) -> (4, Relu) ->  1

## Cifar (kaggle 50,47%)

input is the image (32x32) pixels rgb values 

model 3072 -> (256, Relu) -> (256, Relu) -> 10 :
- 34% 50 epochs
- 50% 700 epochs
- 89% 2300 epochs (overtrained. only 45% on kaggle)

model 34% (75 epoch) : 3072 -> (1024, Relu) -> (256, Relu) -> (64, Relu) -> 10




## Word2Vec

Training set takes the 4 words around the words, for example :

"John is eating the chicken"
we would get for the word eating : 
(eating, John)
(eating, is)
(eating, the)
(eating, chicken)



Testing on 1000 most used words : looking at the most similars words for "game" we get : 
[("game",0.9999999),("i",0.8042298),("fun",0.7797622),("what",0.7150335),("have",0.6954251),("this",0.67785543),("would",0.66231537),("one",0.6454064),("guess",0.64173263),("good",0.62614894)]

creating the training set is long, it takes 4 sec for 1000 elements and 40 for 10000


extracting word list : 
optimization have been made to extract the word list, going froms few hours for 10000 words to 11 sec
```
-- 10 min for 10000 lines
count :: Eq a => a -> [a] -> Int
count x = length . filter (x ==)

-- 2 sec for 10000 lines
countWords :: [B.ByteString] -> M.Map B.ByteString Int
countWords = foldl' (\acc word -> M.insertWith (+) word 1 acc) M.empty
```

try with unboxed data type