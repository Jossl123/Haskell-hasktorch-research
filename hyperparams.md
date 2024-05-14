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

