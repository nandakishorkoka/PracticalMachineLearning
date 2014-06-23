---
title: "assignment.Rmd"
author: "Nandakishor Koka"
date: "June 22, 2014"
output: html_document
---

Initial Setup - Load the Caret package; Make sure all dependencies are installed  a


```r
library(caret) 
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
```{r] 
#fitControl <- trainControl(method = "repeatedcv",number = 10,repeats = 10)
fitControl <- trainControl(number=1)
model = train(classe ~ ., data = training_new, method = "gbm",  verbose = TRUE, trControl = fitControl)
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



