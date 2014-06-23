---
title: "assignment.Rmd"
author: "Nandakishor Koka"
date: "June 22, 2014"
output: html_document
---

Initial Setup - Load the Caret package; Make sure all dependencies are installed  a


```r
library(caret) 
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(9) 
```

Load the data sets 


```r
hlmtrain <- read.csv("data/pml-training.csv", header=TRUE)
hlmtest <- read.csv("data/pml-testing.csv", header=TRUE)
```


Partition the training data for training and cross validation 

```r
inTrain <- createDataPartition (y=hlmtrain$classe, p = 0.80, list=FALSE )
training <- hlmtrain[inTrain,]
testing <- hlmtrain[-inTrain,]
```


Remove variables that have near zero variance 


```r
nzv <- nearZeroVar(hlmtrain)
training <- training[, -nzv] 
testing <- testing[, -nzv]
```


Filter for only numerical columns 

```r
nfs = which(lapply(training,class) %in% c('numeric'))
preproc <- preProcess(training[,nfs], method=c('bagImpute'))
training_new <- cbind(training$classe, predict(preproc, training[,nfs]))
testing_new <- cbind(testing$classe, predict(preproc, testing[,nfs]))
```


Correct the label 

```r
names(training_new)[1] <- "classe"
names(testing_new)[1] <- "classe"
```


Check for variable with high correlation and remove these 

```r
nfs2 = which(lapply(training_new,class) %in% c('numeric'))
dcor <- cor(training_new[,nfs2])
highcor <- findCorrelation(dcor, cutoff = 0.90)
training_new <- training_new[,-highcor]
```


Train the model. Using Gradient boosting as it provides better flexibility in tweaking the model. Limiting the number to allow processing on small machines.  

```r
fitControl <- trainControl(number=1)
model = train(classe ~ ., data = training_new, method = "gbm",  verbose = TRUE, trControl = fitControl)
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
## Loading required package: plyr
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094            -nan     0.1000    0.1235
##      2        1.5260            -nan     0.1000    0.0814
##      3        1.4713            -nan     0.1000    0.0620
##      4        1.4296            -nan     0.1000    0.0523
##      5        1.3931            -nan     0.1000    0.0497
##      6        1.3605            -nan     0.1000    0.0443
##      7        1.3302            -nan     0.1000    0.0377
##      8        1.3063            -nan     0.1000    0.0299
##      9        1.2859            -nan     0.1000    0.0287
##     10        1.2663            -nan     0.1000    0.0290
##     20        1.1302            -nan     0.1000    0.0158
##     40        0.9803            -nan     0.1000    0.0091
##     60        0.8863            -nan     0.1000    0.0072
##     80        0.8167            -nan     0.1000    0.0038
##    100        0.7628            -nan     0.1000    0.0031
##    120        0.7179            -nan     0.1000    0.0026
##    140        0.6789            -nan     0.1000    0.0024
##    150        0.6621            -nan     0.1000    0.0022
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094            -nan     0.1000    0.1748
##      2        1.4957            -nan     0.1000    0.1132
##      3        1.4197            -nan     0.1000    0.1006
##      4        1.3552            -nan     0.1000    0.0769
##      5        1.3048            -nan     0.1000    0.0661
##      6        1.2624            -nan     0.1000    0.0575
##      7        1.2258            -nan     0.1000    0.0485
##      8        1.1934            -nan     0.1000    0.0498
##      9        1.1616            -nan     0.1000    0.0398
##     10        1.1351            -nan     0.1000    0.0383
##     20        0.9458            -nan     0.1000    0.0173
##     40        0.7468            -nan     0.1000    0.0107
##     60        0.6275            -nan     0.1000    0.0089
##     80        0.5389            -nan     0.1000    0.0026
##    100        0.4776            -nan     0.1000    0.0038
##    120        0.4290            -nan     0.1000    0.0042
##    140        0.3877            -nan     0.1000    0.0032
##    150        0.3699            -nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094            -nan     0.1000    0.2077
##      2        1.4729            -nan     0.1000    0.1461
##      3        1.3784            -nan     0.1000    0.1188
##      4        1.3022            -nan     0.1000    0.0947
##      5        1.2422            -nan     0.1000    0.0871
##      6        1.1862            -nan     0.1000    0.0671
##      7        1.1417            -nan     0.1000    0.0659
##      8        1.0998            -nan     0.1000    0.0597
##      9        1.0630            -nan     0.1000    0.0523
##     10        1.0299            -nan     0.1000    0.0516
##     20        0.8079            -nan     0.1000    0.0217
##     40        0.5846            -nan     0.1000    0.0105
##     60        0.4660            -nan     0.1000    0.0055
##     80        0.3903            -nan     0.1000    0.0042
##    100        0.3317            -nan     0.1000    0.0033
##    120        0.2893            -nan     0.1000    0.0039
##    140        0.2548            -nan     0.1000    0.0025
##    150        0.2398            -nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094            -nan     0.1000    0.2098
##      2        1.4767            -nan     0.1000    0.1466
##      3        1.3831            -nan     0.1000    0.1120
##      4        1.3104            -nan     0.1000    0.0865
##      5        1.2534            -nan     0.1000    0.0821
##      6        1.2019            -nan     0.1000    0.0613
##      7        1.1623            -nan     0.1000    0.0709
##      8        1.1176            -nan     0.1000    0.0618
##      9        1.0772            -nan     0.1000    0.0481
##     10        1.0469            -nan     0.1000    0.0563
##     20        0.8153            -nan     0.1000    0.0364
##     40        0.6016            -nan     0.1000    0.0132
##     60        0.4882            -nan     0.1000    0.0069
##     80        0.4085            -nan     0.1000    0.0029
##    100        0.3535            -nan     0.1000    0.0022
##    120        0.3079            -nan     0.1000    0.0020
##    140        0.2730            -nan     0.1000    0.0013
##    150        0.2593            -nan     0.1000    0.0015
```


In-Sample accuracy: 

```r
training_new_pred <- predict(model, training_new) 
print(confusionMatrix(training_new_pred, training_new$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4386  110    6    0    7
##          B   43 2780   95   19   32
##          C   17  114 2558   90   68
##          D   12   20   69 2453   57
##          E    6   14   10   11 2722
## 
## Overall Statistics
##                                         
##                Accuracy : 0.949         
##                  95% CI : (0.945, 0.952)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.936         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.983    0.915    0.934    0.953    0.943
## Specificity             0.989    0.985    0.978    0.988    0.997
## Pos Pred Value          0.973    0.936    0.898    0.939    0.985
## Neg Pred Value          0.993    0.980    0.986    0.991    0.987
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.279    0.177    0.163    0.156    0.173
## Detection Prevalence    0.287    0.189    0.181    0.166    0.176
## Balanced Accuracy       0.986    0.950    0.956    0.971    0.970
```


Out-of-Sample accuracy: 

```r
testing_new_pred <- predict(model, testing_new) 
print(confusionMatrix(testing_new_pred, testing_new$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1093   23    4    0    5
##          B   12  688   22    5   18
##          C    4   34  633   24   21
##          D    6   10   22  605   12
##          E    1    4    3    9  665
## 
## Overall Statistics
##                                         
##                Accuracy : 0.939         
##                  95% CI : (0.931, 0.946)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.923         
##  Mcnemar's Test P-Value : 2.26e-05      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.979    0.906    0.925    0.941    0.922
## Specificity             0.989    0.982    0.974    0.985    0.995
## Pos Pred Value          0.972    0.923    0.884    0.924    0.975
## Neg Pred Value          0.992    0.978    0.984    0.988    0.983
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.279    0.175    0.161    0.154    0.170
## Detection Prevalence    0.287    0.190    0.183    0.167    0.174
## Balanced Accuracy       0.984    0.944    0.950    0.963    0.959
```


Final Test: 

```r
hlmtest <- hlmtest[, -nzv]
final_test <- predict(preproc, hlmtest[,nfs])
final_res <- predict(model, final_test)
print (final_res ) 
```

```
##  [1] C A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



