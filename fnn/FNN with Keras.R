# Packages:
install.packages("caret")
library(caret)
install.packages("keras")
library(keras)
install_keras()

# Data:
load("online.RData") # The data file is not uploaded to github due to confidentiality
str(online)
summary(online)

# Response variable:
y <- ifelse(online$Revenue==TRUE, 1, 0)

# Input selection (removing redundant variables):
cor(online[,-c(11,12,13)]) #BounceRate and ExitRate highly correlated
online <- online[,-7] # Remove BounceRate
online <- online[,-6] # Remove ProductDuration

# Converting inputs to numeric:
online$Weekend <- ifelse(online$Weekend==TRUE, 1, 0)
X <- data.frame(model.matrix(Revenue~., data=online))[,-1]

# Standardize inputs:
X <- data.frame(scale(X))

# Create Validation set:
set.seed(123)
ind <- createDataPartition(online$Revenue, p = .7, list = FALSE)
X.train <- X[ind,]
y.train <- y[ind]
X.test <- X[-ind,]
y.test <- y[-ind]

y.train <- to_categorical(y.train, 2)
colnames(y.train) <- c("FALSE", "TRUE")
y.test <- to_categorical(y.test, 2)
colnames(y.test) <- c("FALSE", "TRUE")
X.train <- as.matrix(X.train)
X.test <- as.matrix(X.test)



# Finding optimal batch size, learning rate and momentum:
batch.size <- c(50, 100, 200)
lr.start <- c(0.01, 0.1, 0.2, 0.3)
momentum <- c(0, 0.3, 0.6, 0.9)

grid <-expand.grid(batch.size = batch.size, 
                   lr.start=lr.start, 
                   momentum=momentum)

f <- integer(nrow(grid)) # vector for storing accuracies

for(i in 1:nrow(grid)) {
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 9, 
              activation = "sigmoid", 
              input_shape = c(11), 
              batch_input_shape=c(grid[i,1], 11)) %>% 
  layer_dense(units = 2, 
              activation = "sigmoid") 
summary(model)

model %>% 
  compile(
    optimizer = optimizer_sgd(lr=grid[i,2], 
                              momentum=grid[i,3], 
                              decay=grid[i,2]/2000),
    loss = 'binary_crossentropy',
    metrics = c('accuracy'))

# Automatically stop training when the validation loss is not decreasing anymore:
callbacks <-  callback_early_stopping(patience = 5, 
                                      monitor = 'val_loss', 
                                      min_delta = 0.01)

history <- model %>% 
  fit(
    X.train, 
    y.train, 
    epochs = 2000, 
    callbacks=callbacks,
    batch_size = grid[i,1],
    validation_data=list(X.test,y.test))
plot(history)
f[i] <- history$metrics$val_accuracy[length(history$metrics$val_accuracy)]
print(f)
}

# Store optimal parameters:
optimal.batch <- grid[which.max(f), 1]
optimal.lr <- grid[which.max(f), 2]
optimal.momentum <- grid[which.max(f), 3]



# Finding optimal learning rate and momentum:
lr.start <- c(0.01, 0.1, 0.2, 0.3)
momentum <- c(0, 0.3, 0.6, 0.9)

grid.lr.momentum <-expand.grid(lr.start = lr.start, momentum=momentum)

f <- integer(nrow(grid.lr.momentum)) # vector for storing accuracies

for(i in 1:nrow(grid.lr.momentum)) {
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = 9, 
                activation = "sigmoid", 
                input_shape = c(11), 
                batch_input_shape=c(optimal.batch, 11)) %>% 
    layer_dense(units = 2, 
                activation = "sigmoid") 
  summary(model)
  model %>% 
    compile(
      optimizer = optimizer_sgd(lr=grid.lr.momentum[i, 1], 
                                momentum=grid.lr.momentum[i, 2], 
                                decay=grid.lr.momentum[i,1]/optimal.epochs),
      loss = 'binary_crossentropy',
      metrics = c('accuracy'))
  history <- model %>% 
    fit(
      X.train, 
      y.train, 
      epochs = optimal.epochs, 
      batch_size = optimal.batch,
      validation_data=list(X.test,y.test))
  plot(history)
  f[i] <- history$metrics$val_accuracy[length(history$metrics$val_accuracy)]
  print(f)
}

# Store optimal parameters:
optimal.lr <- grid.lr.momentum[which.max(f), 1]
optimal.momentum <- grid.lr.momentum[which.max(f), 2]


# Optimal model:
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 9, 
              activation = "sigmoid", 
              input_shape = c(11), 
              batch_input_shape=c(optimal.batch, 11)) %>% 
  layer_dense(units = 2, 
              activation = "sigmoid") 
summary(model)
model %>% 
  compile(
    optimizer = optimizer_sgd(lr=optimal.lr,
                              momentum=optimal.momentum, 
                              decay=optimal.lr/2000),
    loss = 'binary_crossentropy',
    metrics = c('accuracy'))

callbacks <-  callback_early_stopping(patience = 20, monitor = 'val_loss')

history <- model %>% 
  fit(
    X.train, 
    y.train, 
    callbacks = callbacks,
    epochs = 2000,
    batch_size = optimal.batch,
    validation_data=list(X.test,y.test))
plot(history, xlim=c(0,31))
history$metrics



# Accuracy:
evaluation <- model %>% 
  evaluate(
    X.test, 
    y.test, 
    batch_size=optimal.batch)
evaluation
