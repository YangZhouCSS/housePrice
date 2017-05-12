#clear environment
rm(list=ls())

library(ggplot2)
library(glmnet)
library(Metrics)
library(gbm)
library(randomForest)
library(dplyr)
library(corrplot)



#set directory
setwd("C:/Users/X1/Desktop/stat515/project/data")

##data preprocessing
train <- read.csv("train.csv",header = TRUE)
test <- read.csv("test.csv",header = TRUE)


#correlation plots
vtrain <- train[ , sapply(train, is.numeric)] #only consider numeric variables
M=cor(vtrain)

corrplot(M, method = "number")
corrplot(M, method = "circle")

top1 = vtrain$OverallQual
top2 = vtrain$GrLivArea
top3 = vtrain$GarageCars
top4 = vtrain$GarageArea

pairs(~SalePrice+OverallQual+GrLivArea+GarageCars+GarageArea,data=vtrain,
      main="Scatterplot Matrix")

#combine train and test for preprocessing
train0 = train
train$SalePrice = NULL

all_data <- rbind(select(train,MSSubClass:SaleCondition),
                   select(test,MSSubClass:SaleCondition))

# delete columns with missing data
all_data$PoolQC=NULL
all_data$MiscFeature=NULL
all_data$Alley=NULL
all_data$Fence=NULL
all_data$FireplaceQu=NULL
all_data$LotFrontage=NULL
all_data$GarageCond=NULL
all_data$GarageFinish=NULL
all_data$GarageQual=NULL
all_data$GarageType=NULL
all_data$GarageYrBlt=NULL
all_data$BsmtExposure=NULL
all_data$BsmtFinType2=NULL
all_data$BsmtCond=NULL
all_data$BsmtFinType1=NULL
all_data$BsmtQual=NULL
all_data$MasVnrArea=NULL
all_data$MasVnrType=NULL


#create dummies
library(dummies)
dummy.all <- dummy.data.frame(all_data, sep="_")


#seperate train and test
X_train <- dummy.all[1:nrow(train),]
X_test <- dummy.all[(nrow(train)+1):nrow(all_data),]
y <- train0$SalePrice

#write the transformed data into csv
X_train_w = X_train
X_train_w[["SalePrice"]] <- y
write.csv(X_train_w, file = "Mytrain.csv")
write.csv(X_test, file = "Mytest.csv")

#log-transformation to get less skewed data
df <- rbind(data.frame(version="log(price)",x=log(y )),
            data.frame(version="price",x=y))

y <- log(y)

#comparison before and after log-transformation
ggplot(data=df) +
  facet_wrap(~version,ncol=2,scales="free_x") +
  geom_histogram(aes(x=x))

#df to matrix
X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)

###model1-Ridge
lambdas <- seq(1,0,-0.001)

#cross validation to get the best lambda
set.seed(12345)
cv.out = cv.glmnet(X_train,y,alpha=0,nfolds=10,lambda=lambdas)
plot(cv.out)
bestlam = cv.out$lambda.min
bestlam  #0.119

#build model and predict test set
ridge.model = glmnet(X_train,y,alpha=0,lambda=bestlam)
ridge.pred=predict(ridge.model,s=bestlam,newx=X_test)
preds <- exp(ridge.pred) 
m = toString(mean(preds,na.rm=TRUE))  #178089.36
write.csv(preds, file = "results_ridge.csv",na=m) #replace NA with the average sale price
###score/ rmse = 0.13531 on test set

###model2-Lasso
lambdas <- seq(1,0,-0.001)

#cross validation to get the best lambda
set.seed(12345)
cv.out = cv.glmnet(X_train,y,alpha=1,nfolds=10,lambda=lambdas)
plot(cv.out)
bestlam = cv.out$lambda.min
bestlam  #0.004

#predict
lasso.model = glmnet(X_train,y,alpha=1,lambda=bestlam)
lasso.pred=predict(lasso.model,s=bestlam,newx=X_test)
preds <- exp(lasso.pred)
m = toString(mean(preds,na.rm=TRUE))  
write.csv(preds, file = "results_Lasso.csv",na=m)
###score/ rmse = 0.12967 on test set

###model3-randomForest
X_train <- all_data[1:nrow(train),]
X_test <- all_data[(nrow(train)+1):nrow(all_data),]
X_train[["SalePrice"]] <- exp(y)
X_train <- X_train[ , apply(X_train, 2, function(x) !any(is.na(x)))]

rf = randomForest(SalePrice~., data=X_train, importance=TRUE)

importance(rf)
varImpPlot(rf)

#predict
tree.pred = predict(rf, X_test)
m = toString(mean(tree.pred,na.rm=TRUE))
write.csv(tree.pred, file = "results_RF.csv",na=m)
###score/ rmse = 0.14641 on test set



