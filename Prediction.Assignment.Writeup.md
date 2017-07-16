**Prediction Assignment Writeup**

by Arif A. Arshad

*Note: complete code can be found in the RMarkdown file.  The preference of the instructors seemed to be to heed "security concerns with the exchange of R code." (Info page for Prediction Assignment Writeup)*

**1. Executive Summary**

The report describes how I constructed two highly accurate predictive models for an exercise classification problem and quiz.  I describe in detail how I cleaned the data, how I chose predictors, how I used cross validation, and how I was able to get an out of sample error rate for the models.  I explain each of my choices in the model building process in detail.  Importantly, both models had high accuracy rates.  Random forest was more accurate than gbm at 99% versus 96%.  However, both provided the same predictions for the test set.

**2. Data**

  ***The Dataset:  What is it about?  Where is it from?***  
     
The data comes from the following study and the dataset can be found at the site cited:

  *Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.*
    
 *Read more: http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises#ixzz4mwey0LP3*

The train and test datasets provided for the assignment consist of data on how well "six young, healthy" participants, ranging in ages from 20 to 28, with no weightlifting experience, performed dumbbell curls--10 repetitions each--in 5 different ways.  Body movements while performing the barbell lifts were recorded from accelerometers on the belt, forearm, arm and dumbell of the 6 participants.  They could perform the barbell lifts incorrectly or correctly.  The first of the five factor levels corresponds to a correct curl done in the prescribed manner, while the other four correspond to common mistakes in performing dumbbell lifts.  I wanted to predict the quality of the dumbbell curl and if done incorrectly, to identify the mistake.

The quality of the dumbbell curl is the categorical response variable "classe" which represents the way in which the six subjects performed the lifts .  They are specified as follows:

  "A": Curl performed correctly.  Done as specified.
  
  "B": Curl performed incorrectly.  Throwing elbows to the front.
  
  "C": Curl performed incorrectly.  Lifting the dumbbell only halfway.
  
  "D": Curl performed incorrectly.  Lowering the dumbbell only halfway.
  
  "E": Curl performed incorrectly.  Throwing the hips to the front.



  ***Cleaning the Data:  How was it dirty?  How was it cleaned up?***

The data was dirty in a few ways, much of which I learned with trial and error.  It contained many columns that were completely NAs.  Moreover, the test set contained more columns with NAs than the train set.  The models I chose--random forest and gbm--will fail with the NAs.  As a result, the first task was to eliminate NA columns using is.na() and the resulting indices.  I also had to make sure that the number of columns in the train and test set were equal.  Since the test set contained more NA columns, I used the indices that identified NAs from the test to eliminate the same columns from the train and test.  After removing NA columns, I was left with 59 independent variables.

Second, I needed to remove any predictor variables that had practically no variance.  I used caret::nearZeroVar() with saveMetrics set to FALSE. This way an index is returned which can then be used to trim the dataset.  

There was only one variable that had near zero variance.  It was the "new_window" variable.

Third, I learned after repeatedly overfitting my models and getting odd predictions, that I needed to remove the id variables.  These included "X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", and "num_window".  These numeric vectors were responsible for overfitting. 

After the process of manual elimination, I was left with 53 predictor variables.

I will discuss data partitioning and cross validation after explaining my model choices below.



**3. Model Selection**

  ***Model Kind:  Why did I choose certain algorithms?***  

My original intention was to combine a random forest and gbm model.  However since both eventually became independently reliable, I did not feel the need to combine. I chose to construct two models because I wanted to see if they would give me the same predictions for the quiz. 

I chose to use random forest and gbm for two reasons.  First, the response variable is a factor.  Hence, the problem is one of classification.  Second, random forest and gbm are known to be two of the most powerful algorithms for prediction.  The aim of this assignment is predictive accuracy, not interpretability.  As a result, random forest and gbm models were more than adequate to the task.
        
***Building the Model:  Designing the Model***
    
  ***Data Partition: How and Why?  Cross-Validation***
          
The given test set did not contain the values for response due to the quiz portion of the course project.  This meant that I needed a way to get an estimate of the out of sample error before I applied to the model to quiz.  For these aims, I partitioned the train data into a train set and a validation set.  

This was one of the two ways I used cross validation.

  ***Choosing Predictors: How?*** 

As I explained above, I chose predictors manually by elimination.  I eliminated any predictor that made the random forest and gbm algorithms run awry.  Therefore, I ruled out variables that consisted of NAs, had zero variance, or were id numeric vectors. I did not have to make more complicated choices about removing NAs because the various columns were either devoid of NAs or completely filled with them.

Beyond manual elimination, the random forest and gbm algorithms chose an ensemble of predictors programmatically.

  ***Data Processing Speed for Random Forest and Boost***

One of the problems I initially faced was the processing speed for these algorithms using the train function in caret.  The processing was extremely slow and gave empty results (NAs).  I then read our mentor Len's article on how to setup the train function for parallel processing which markedly improved the speed and performance of the algorithms. 

The code involved setting up clusters with the parallel and doParallel packages and then configuring a train control object as an argument within the train function.  


```r
## configure parallel processing
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
## configure trainControl object
fitControl <- trainControl(method="cv", number=10, allowParallel = TRUE)
```

Through the train control object, I was able to use **k-folds cross validation**.  This was the second way I used cross validation.  **Both uses of cross validation (i.e. partitioning the training set to create a validation set and using k-folds) worked on the problem of overfitting.  In the former, it made the problem tractable by allowing estimation of out of sample error on an independent data set.  In the latter, the chances of overfitting were reduced by averaging across folds**.
        
**4.  Model Evaluation**

The random forest and gbm models and their results, including measures of out of sample error from the validation set are below.


```r
# the models
fit_rf <- train(classe~., method="rf", data=train_train, trControl=fitControl)
fit_boost <- train(classe~., method="gbm", data=train_train, trControl=fitControl,verbose=FALSE)
stopCluster(cluster)
registerDoSEQ()

# predictions on validation set for out of sample error measures
pred_rf <- predict(fit_rf, train_val)
pred_boost <- predict(fit_boost, train_val)
confusionMatrix(pred_rf, train_val$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1668   10    0    0    0
##          B    5 1126   10    1    1
##          C    1    3 1014   15    3
##          D    0    0    2  946    1
##          E    0    0    0    2 1077
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9908         
##                  95% CI : (0.988, 0.9931)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9884         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9886   0.9883   0.9813   0.9954
## Specificity            0.9976   0.9964   0.9955   0.9994   0.9996
## Pos Pred Value         0.9940   0.9851   0.9788   0.9968   0.9981
## Neg Pred Value         0.9986   0.9973   0.9975   0.9964   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2834   0.1913   0.1723   0.1607   0.1830
## Detection Prevalence   0.2851   0.1942   0.1760   0.1613   0.1833
## Balanced Accuracy      0.9970   0.9925   0.9919   0.9904   0.9975
```

```r
confusionMatrix(pred_boost, train_val$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1651   41    0    2    1
##          B    9 1073   33    5    9
##          C   13   23  982   36    9
##          D    1    1   10  914   11
##          E    0    1    1    7 1052
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9638          
##                  95% CI : (0.9587, 0.9684)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9542          
##  Mcnemar's Test P-Value : 1.258e-10       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9863   0.9421   0.9571   0.9481   0.9723
## Specificity            0.9896   0.9882   0.9833   0.9953   0.9981
## Pos Pred Value         0.9740   0.9504   0.9238   0.9755   0.9915
## Neg Pred Value         0.9945   0.9861   0.9909   0.9899   0.9938
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2805   0.1823   0.1669   0.1553   0.1788
## Detection Prevalence   0.2880   0.1918   0.1806   0.1592   0.1803
## Balanced Accuracy      0.9879   0.9651   0.9702   0.9717   0.9852
```
   
Both models have low error rates, with **random forest having an out of sample error rate of about one percent across the various measures used to estimate error including sensitivity, specificity, accuracy, and concordance**.  I cannot show the predictions on the test set since it was part of the quiz.  However, both models gave the same test set predictions.

