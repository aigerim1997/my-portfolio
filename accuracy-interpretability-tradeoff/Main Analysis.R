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

# Load the data:
load("Star Wars Clean.RData")
summary(data.clean)
str(data.clean)


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
                                         "Graduate degree"), 
                               ordered = TRUE)


# Data preparation for logistic regression:
## Function to convert ordinal factor to numeric:
convert <- function(x) {
  as.numeric(levels(x))[x]
}

## Convert Age, Household.Income to numeric:
levels(data.logit$Age) <- c(1,2,3,4)
data.logit$Age <- convert(data.logit$Age)

levels(data.logit$Household.Income) <- c(1,2,3,4,5)
data.logit$Household.Income <- convert(data.logit$Household.Income)

levels(data.logit$Education) <- c("Bachelor Or Associate", 
                                  "Graduate", 
                                  "High School",
                                  "High School",
                                  "Bachelor Or Associate") 
data.logit$Education <- factor(data.logit$Education, 
                               levels = c("High School", 
                                          "Bachelor Or Associate", 
                                          "Graduate"),
                               ordered = TRUE)
levels(data.logit$Education) <- c(1,2,3)
data.logit$Education <- convert(data.logit$Education)

## Checking for multicollinearity (for logistic regression) by computing correlations:
cor(data.logit[, -c(1,22:23, 27)])

## Visualize the correlations:
corrplot(cor(data.logit[, -c(1, 22:23, 27)]), 
         method = "color", 
         tl.cex=0.8, 
         number.cex = 0.5)

## Predictors and response variable:
y <- ifelse(data.logit$fan.StarTrek=="Yes", 1, 0)
X <- model.matrix(fan.StarTrek~. ,data=data.logit)[, -1]

## Standardize non-factor variables:
X.logit <- X
X.logit[, -c(1, 22, 26:34)] <- scale(X[, -c(1, 22, 26:34)])  

# Dividing the data into training and test sets 70/30%:
set.seed(256) 
ind <- createDataPartition(data.clean$fan.StarTrek, p=0.7, list=FALSE)  

y.train <- y[ind]
X.train.logit <- X.logit[ind,]
data.train.rf <- data.clean[ind,]

y.test <- y[-ind]
X.test.logit <- X.logit[-ind,]
data.test.rf <- data.clean[-ind,]


# Run logit with LASSO: 
## Cross-validation to choose optimal lambda:
set.seed(123)
logit.cv <- cv.glmnet(X.train.logit, y.train, alpha=1 , family = "binomial",
                      lambda = 10^seq(-3, 0.1, length.out = 80), 
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





# Random Forest:
## Choosing the number of trees:
set.seed(113)
rf <- randomForest(fan.StarTrek ~., 
                   data = data.train.rf, 
                   mtry = 26, 
                   ntree = 3000)

## Plot the process of tuning the Random forest:
par(mfrow= c(1,2),
    mgp=c(1.3,0.7,0), 
    oma=c(0,0,0,0) )

plot(rf$err.rate[,1], type="l", 
     xlab = "Number of Trees", 
     ylab= "OOB error", 
     main = "a) Tuning Number of Trees", 
     cex.main=0.5, 
     cex.lab=0.5, 
     cex.axis=0.5)   # Stabilizes after 2300

## Tuning mtry:
oob.error <- double(26) # Vector to store OOB errors
m <- seq(1, 26, by=1)
for(mtry in 1:26) {
  rf <- randomForest(fan.StarTrek ~., 
                     data = data.train.rf, 
                     mtry = mtry, 
                     ntree = 2500) # Optimal number of trees
  oob.error[mtry] <- mean(rf$err.rate[,1])
  cat(oob.error[mtry], " ")
}

which.min(oob.error) # Display the optimal mtry hyperparameter

plot(m, oob.error, type="l",
     xlab="Number of variables",
     ylab="OOB error", 
     main="b) Tuning Number of Variables Considered at Each Split", 
     cex.main=0.5, 
     cex.lab=0.5, 
     cex.axis=0.5) # 7


## Optimal Random Forest:
set.seed(123)
rf.best <- randomForest(fan.StarTrek ~., 
                        data = data.train.rf, 
                        mtry = which.min(oob.error), 
                        ntree = 2500,
                        importance=TRUE)
pred.rf <- predict(rf.best, 
                   newdata=data.test.rf, 
                   type="class")

## Accuracy of the Logit model (LASSO):
confusionMatrix(data=pred.test.logit, 
                reference = y.test) # 0.6813 accuracy

## Accuracy of the random forest model:
confusionMatrix(data=pred.rf, 
                reference = y.test) # 0.6978 accuracy



# Interpretation:

## Variable Importance Plot for random forest:
par(mfrow=c(1,1), mgp=c(2,1,0))
set.seed(155)
varImpPlot(rf.best, 
           type=1, 
           main="Variable Importance (Measured by Decrease in Accuracy after Permutation)", 
           cex=0.7 )


dotchart(sort(importance(rf.best, scale=FALSE)[,1]), cex=0.7)
importance(rf.best, scale=FALSE)[]
varImp(rf.best)


## Partial Dependence Plots for random forest:
install.packages("iml")
library(iml)

model <- Predictor$new(rf.best, 
                       data=data.train.rf, 
                       type="prob", 
                       class="Yes")
par(mfrow=c(2,2))

# PDP for fan.StarWars:
eff <- FeatureEffect$new(model, 
                         feature="fan.StarWars" ,  
                         method="pdp")
Plot1 <-plot(eff) +
  labs(title="a) fan.StarWars") + 
  theme(title=element_text(size=8))


# PDP for EmperorPalpatine:
eff <- FeatureEffect$new(model, 
                         feature="EmperorPalpatine",  
                         method="pdp", 
                         grid.size = 5)
Plot2 <- plot(eff) +  
  labs(title="b) EmperorPalpatine") + 
  theme(title=element_text(size=8))


# PDP for Padme Amidala:
eff <- FeatureEffect$new(model, feature="PadmeAmidala" ,  
                         method="pdp", 
                         grid.size = 5)
Plot3 <- plot(eff) +  
  labs(title="d) PadmeAmidala") + 
  theme(title=element_text(size=8))

## Display the PDP plots of the most important 4 variables:
grid.arrange(Plot1, Plot2, Plot4, Plot3, nrow=2)

