# Packages:
install.packages("corrplot")
install.packages("glmnet")
install.packages("rpart")
install.packages("caret")
install.packages("randomForest")
install.packages("e1071")


library(glmnet)
library(corrplot)
library(rpart)
library(caret)
library(randomForest)
library(e1071)

load("Star Wars Clean.RData")
summary(data.clean)

# Data preparation for Random Forest:
## Convert Age, Education and Income to ordinal:
data.clean$Age <- factor(data.clean$Age, 
                         levels=c("18-29", "30-44", "45-60", "> 60"), 
                         ordered=TRUE)
data.clean$Household.Income <- factor(data.clean$Household.Income, 
                                      levels=c("$0 - $24,999",
                                               "$25,000 - $49,999", 
                                               "$50,000 - $99,999",
                                               "$100,000 - $149,999",
                                               "$150,000+"), 
                                      ordered=TRUE)
data.logit <- data.clean

data.clean$Education <- factor(data.clean$Education, 
                               levels= c("Less than high school degree", 
                                         "High school degree", 
                                         "Some college or Associate degree",
                                         "Bachelor degree", 
                                         "Graduate degree"), ordered = TRUE)


# Data preparation for logit:
## Function to convert ordinal factor to numeric:
convert <- function(x) {
  as.numeric(levels(x))[x]}

## Convert Age, Household.Income to numeric:
levels(data.logit$Age) <- c(1,2,3,4)
data.logit$Age <- convert(data.logit$Age)

levels(data.logit$Household.Income) <- c(1,2,3,4,5)
data.logit$Household.Income <- convert(data.logit$Household.Income)

## Merging levels of Education:
levels(data.logit$Education) <- c("Bachelor Or Associate", 
                                  "Graduate", 
                                  "High School",
                                  "High School",
                                  "Bachelor Or Associate") 

## Checking for multicollinearity (logit):
cor(data.logit[, -c(1,22:23, 26:27)])
corrplot(cor(data.logit[, -c(1, 22:23, 26:27)]), method = "color", tl.cex=0.8, number.cex = 0.5   )

## Divide into predictors and response:
y <- ifelse(data.logit$fan.StarTrek=="Yes", 1, 0)
X <- model.matrix(fan.StarTrek~. ,data=data.logit)[, -1]

# Standardize non-factor variables:
X.logit <- X
X.logit[, -c(1, 22, 25:34)] <- scale(X[, -c(1, 22, 25:34)])  


# Run logit with LASSO: 
## Data preparation and creating CV partitions:
set.seed(1000) 
ind <- createFolds(data.clean$fan.StarTrek, k=5) 

## Cross-validation to choose optimal lambda:
cv.accuracy <- double(5)
for(i in 1:5) {
  index <- eval(parse(text = paste0("ind$Fold", i, sep="")))
  y.train <- y[-index]
  X.train.logit <- X.logit[-index,]
  
  y.test <- y[index]
  X.test.logit <- X.logit[index,]
  
  set.seed(123)
  logit.cv <- cv.glmnet(X.train.logit, y.train, alpha=1 , family = "binomial",
                        lambda = 10^seq(-3, 0.2, length.out = 50), 
                        standardize = FALSE, nfolds=10, 
                        type.measure = "class")
  plot(logit.cv)
  logit.cv$lambda.min
  
  ## Optimal model:
  logit.best <- glmnet(X.train.logit, y.train, alpha = 1, family = "binomial", 
                       lambda = logit.cv$lambda.min, standardize = FALSE)
  round(logit.best$beta, digits=3) # Coefficients
  
  ## Accuracy of logit:
  pred.test.logit <- predict(logit.best, newx = X.test.logit)
  pred.test.logit <-factor(ifelse(pred.test.logit>0.5, "Yes", "No"))
  table(pred.test.logit, y.test)
  y.test <- factor(ifelse(y.test==1, "Yes", "No"))
  cm <- confusionMatrix(data=pred.test.logit, reference = y.test)
  cv.accuracy[i] <-round(cm$overall[1], digits=4)
}
mean(cv.accuracy)  # 5-fold cv accuracy
cv.accuracy
range(cv.accuracy)



# Random Forest:
## Choosing the number of trees:
cv.accuracy.rf <- double(5)
for(i in 1:5) {
  index <- eval(parse(text = paste0("ind$Fold", i, sep="")))
  data.train.rf <- data.clean[-index,]
  data.test.rf <- data.clean[index,]
  y.test <- y[index]
  rf <- randomForest(fan.StarTrek ~., data = data.train.rf, mtry = 26, ntree = 1000)
  plot(rf$err.rate[,1], type="l", 
       xlab = "Number of Trees", 
       ylab= "OOB error")   # Stabilizes after 2000
  
  ## Tuning mtry:
  oob.error <- double(26) # Vector to store OOB errors
  m <- seq(1, 26, by=1)
  for(mtry in 1:26) {
    rf <- randomForest(fan.StarTrek ~., data = data.train.rf, mtry = mtry, ntree = 800)
    oob.error[mtry] <-mean(rf$err.rate[,1])
    cat(oob.error[mtry], " ")
  }
  
  which.min(oob.error)
  
  plot(m, oob.error, type="l",
       xlab="Number of variables",
       ylab="OOB error") # 7
  
  ## Optimal Random Forest:
  rf.best <- randomForest(fan.StarTrek ~., data = data.train.rf, 
                          mtry = which.min(oob.error), ntree = 800,
                          importance=TRUE)
  pred.rf <- predict(rf.best, newdata=data.test.rf, type="class")
  y.test <- factor(ifelse(y.test==1, "Yes", "No"))
  cm.rf <- confusionMatrix(data=pred.rf, reference = y.test)
  cv.accuracy.rf[i] <-round(cm.rf$overall[1], digits=15)
}


mean(cv.accuracy.rf)
range(cv.accuracy.rf)
mean(cv.accuracy)
range(cv.accuracy)

