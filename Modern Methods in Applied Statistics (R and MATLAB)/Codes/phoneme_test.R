library(bootstrap)
library(earth)
library(rpart.plot)
library(rpart)
library(rgl)
library(reshape)
library(car)
library(glmnet)
library(BMA)
library(mgcv)
library(nnet)
library(MASS)
library(randomForest)
library(gbm)

setwd('d:/STAT852/project2')

phoneme <-  read.table("phoneme.csv", header=FALSE, sep=",", na.strings=" ", skip = 1)
phoneme <- phoneme[,c(2:258)]
p <- ncol(phoneme)-1

rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}



iter <- 5
num.model <- 2

MSE.matrix <- matrix(data=NA,ncol=num.model,nrow=iter)
MSPE.matrix <- matrix(data=NA,ncol=num.model,nrow=iter)
val.err.mat<-matrix(data=NA,nrow=iter,ncol=num.model)
nvar.max <- c(floor(p/3),floor(2*p/3),p)

#ind.matrix <-  read.table("resample_20.csv", header=FALSE, sep=",", na.strings=" ", skip = 1)
ind.matrix <- matrix(data=NA,ncol=iter, nrow=nrow(phoneme))
set.seed(120401002)
for (i in c(1:iter)){
  ind.matrix[,i]<-sample.int(nrow(phoneme), size = nrow(phoneme), replace = TRUE)
}
for (i in c(1:iter)){
  train.data.ind<-ind.matrix[,i]
  test.data.ind<-setdiff(c(1:nrow(phoneme)), unique(train.data.ind))
  train.data<-phoneme[train.data.ind,]
  test.data<-phoneme[test.data.ind,]
  train.data.scaled<-train.data
  train.data.scaled[,c(1:256)] <- rescale(train.data[,c(1:256)],train.data[,c(1:256)])
  test.data.scaled<-test.data
  test.data.scaled[,c(1:256)] <- rescale(test.data[,c(1:256)],train.data[,c(1:256)])
  
  
    # LDA
    lda.fit <- lda(x=train.data[,c(1:256)], grouping=train.data[,257])
    pred.lda.tr <- predict(lda.fit, newdata=train.data[,c(1:256)])
    pred.lda.te <- predict(lda.fit, newdata=test.data[,c(1:256)])
    lda.pred.train <- pred.lda.tr$class
    lda.pred.test <- pred.lda.te$class
    MSE.matrix[i,1] <- mean(ifelse(lda.pred.train == train.data[,257], yes=0, no=1))
    MSPE.matrix[i,1] <- mean(ifelse(lda.pred.test == test.data[,257], yes=0, no=1))
    
    # QDA
    qda.fit <- qda(x=train.data[,c(1:256)], grouping=train.data[,257])
    pred.qda.tr <- predict(qda.fit, newdata=train.data[,c(1:256)])
    pred.qda.te <- predict(qda.fit, newdata=test.data[,c(1:256)])
    qda.pred.train <- pred.qda.tr$class
    qda.pred.test <- pred.qda.te$class
    MSE.matrix[i,2] <- mean(ifelse(qda.pred.train == train.data[,257], yes=0, no=1))
    MSPE.matrix[i,2] <- mean(ifelse(qda.pred.test == test.data[,257], yes=0, no=1))
    
#     # Multinomial Logistic
#     logistic.fit <- multinom(data=train.data.scaled, formula=V258 ~ ., maxit=2000, MaxNWts = 10000)
#     pred.logistic.tr <- predict(logistic.fit, newdata=train.data.scaled)
#     pred.logistic.te <- predict(logistic.fit, newdata=test.data.scaled)
#     MSE.matrix[i,3] <- mean(ifelse(pred.logistic.tr == train.data.scaled$V258, yes=0, no=1))
#     MSPE.matrix[i,3] <- mean(ifelse(pred.logistic.te == test.data.scaled$V258, yes=0, no=1))
    
#     # LASSO
#     model.lasso<-cv.glmnet(y=as.matrix(train.data[,257]), x= as.matrix(train.data[,c(1:256)]), 
#                            family="multinomial",nfolds=5,lambda=c(0.1,0.06,0.03,0.01,0.006))
#     pred.lasso.tr<-predict(model.lasso, newx=as.matrix(train.data[,c(1:256)]),s="lambda.min",type = "class")
#     pred.lasso.te<-predict(model.lasso, newx=as.matrix(test.data[,c(1:256)]),s="lambda.min",type = "class")
#     MSE.matrix[i,4]<-mean(pred.lasso.tr!=train.data[,257])
#     MSPE.matrix[i,4]<-mean(pred.lasso.te!=test.data[,257])
  #   
  #   model.fulltree <- rpart(data=train.data, V258 ~ ., method="class")
  #   pred.ft.tr <- predict(model.fulltree, newdata=train.data, type="class")
  #   pred.ft.te <- predict(model.fulltree, newdata=test.data, type="class")
  #   MSE.matrix[i,5] <- mean(ifelse(pred.ft.tr == train.data[,257], yes=0, no=1))
  #   MSPE.matrix[i,5] <- mean(ifelse(pred.ft.te == test.data[,257], yes=0, no=1))
  #   
  #   p.rpart <- model.fulltree$cptable
  #   xstd <- p.rpart[, 5L]
  #   xerror <- p.rpart[, 4L]
  #   nsplit <- p.rpart[, 2L]
  #   ns <- seq_along(nsplit)
  #   cp0 <- p.rpart[, 1L]
  #   cp <- sqrt(cp0 * c(Inf, cp0[-length(cp0)]))
  #   minpos <- min(seq_along(xerror)[xerror == min(xerror)])
  #   th <- (xerror + xstd)[minpos]
  #   ind <- which(xerror<=th)[1]
  #   cp.1se <- cp[ind]
  #   cp.min <- cp[which.min(xerror)]
  #   
  #   model.1se.tree <- prune(model.fulltree, cp=cp.1se)
  #   model.min.tree <- prune(model.fulltree, cp=cp.min)
  #   
  #   pred.1set.tr <- predict(model.1se.tree, newdata=train.data, type="class")
  #   pred.1set.te <- predict(model.1se.tree, newdata=test.data, type="class")
  #   MSE.matrix[i,6] <- mean(ifelse(pred.1set.tr == train.data[,257], yes=0, no=1))
  #   MSPE.matrix[i,6] <- mean(ifelse(pred.1set.te == test.data[,257], yes=0, no=1))
  #   
  #   pred.mint.tr <- predict(model.min.tree, newdata=train.data, type="class")
  #   pred.mint.te <- predict(model.min.tree, newdata=test.data, type="class")
  #   MSE.matrix[i,7] <- mean(ifelse(pred.mint.tr == train.data[,257], yes=0, no=1))
  #   MSPE.matrix[i,7] <- mean(ifelse(pred.mint.te == test.data[,257], yes=0, no=1))
  #   
  #   model.rf <- randomForest(data=train.data, V258~., ntree=2000, mtry=85, keep.forest=TRUE, nodesize= 100)
  #   pred.rf.tr <- predict(model.rf, newdata=train.data, type="response")
  #   pred.rf.te <- predict(model.rf, newdata=test.data, type="response")
  #   MSE.matrix[i,8] <- mean(ifelse(pred.rf.tr == train.data[,257], yes=0, no=1))
  #   MSPE.matrix[i,8] <- mean(ifelse(pred.rf.te == test.data[,257], yes=0, no=1)) 
  
#   # Boosting Trees
#   model.bt <- gbm(data=train.data, V258~., distribution="multinomial", verbose = FALSE, cv.folds=0)
#   pred.bt.train <- predict(model.bt, newdata=train.data,n.trees=100, type="response")
#   pred.bt.test <- predict(model.bt, newdata=test.data,n.trees=100, type="response")
#   
#   pred.class.train <- apply(pred.bt.train[,,1], 1, which.max)
#   pred.class.test <- apply(pred.bt.test[,,1], 1, which.max)
#   MSE.matrix[i,1] <- mean(ifelse(pred.class.train == as.numeric(train.data[,257]), yes=0, no=1))
#   MSPE.matrix[i,1] <- mean(ifelse(pred.class.test == as.numeric(test.data$V258), yes=0, no=1))
#   print(i)
#   print(MSPE.matrix[i,1])
}

# test_err_frame <- as.data.frame(sqrt(MSPE.matrix))
# 
# names(test_err_frame)<- c("LDA","QDA","Mul.Logit","LASSO","NB","RF","TuGBM","TuSVM","NN")
# # c("LDA","QDA","Mul.Logit","LASSO","FullTree","PrunedTree.1se","PrunedTree.min","R.F.")
# 
# rowMins <- apply(MSPE.matrix,1,min)
# minMatrix <- matrix(data=rep(rowMins,times=9),nrow=5)
# ratio_err_frame <- as.data.frame(sqrt(MSPE.matrix/minMatrix))
# 
# names(ratio_err_frame)<-c("LDA","QDA","Mul.Logit","LASSO","NB","RF","TuGBM","TuSVM","NN")
# 
# win.graph(h=7, w=10, pointsize=12)
# boxplot(test_err_frame,main="test-MSPEs vs Models",ylab="root-MSPE")
# 
# win.graph(h=7, w=10, pointsize=12)
# boxplot(ratio_err_frame,main="root ratios of MSPE to minMSPE vs Models",ylab="ratio")

# MSPE.matrix<-read.table("allResults.csv", header=FALSE, sep=",", na.strings=" ", skip = 0)
# win.graph(h=7, w=10, pointsize=12)
# boxplot(allMSPEs.df)