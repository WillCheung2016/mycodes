library(matrixcalc)
require(plyr)
require(abind)
library(cpca)
library(pcaPP)
library(MASS)

setwd('d:/STAT852/project2')

phoneme <-  read.table("phoneme.csv", header=FALSE, sep=",", na.strings=" ", skip = 1)
phoneme <- phoneme[,c(2:258)]
p <- ncol(phoneme)-1

phoneme[,'V258']<-as.numeric(phoneme[,'V258'])
#phoneme[,'V258']<-as.factor(phoneme[,'V258'])

classify <- function(x,mu,inv.cov,pk){
  loglik.vec<-rep(0,times=5)
  for(i in c(1:5)){
    x<-as.numeric(x)
    m<-as.numeric(mu[i,])
    dif<-matrix(data=x-m,ncol=1)
    loglik.vec[i] <- (-1/2)*t(dif)%*%inv.cov[,,i]%*%(dif) + log(pk[i])
  }
  lab<-which.max(loglik.vec)
  lab
}

predict.qda<-function(data,mu,inv.cov,pk){
  pred<-rep(0,times=nrow(data))
  for(i in c(1:nrow(data))){
    pred[i]<-classify(data[i,],mu,inv.cov,pk)
  }
  pred
}

computeMoments1<-function(data){
  cov.matrices<-array(0,dim=c(256,256,5))
  invcov.matrices<-array(0,dim=c(256,256,5))
  mu.matrix<-array(0,dim=c(5,256))
  
  for(i in c(1:5)){
    cov.matrices[,,i]<-cov(data[which(data$V258==i),c(1:256)])
    invcov.matrices[,,i]<-matrix.inverse(cov.matrices[,,i])
    mu.matrix[i,]<-apply(data[which(data$V258==i),c(1:256)],2,mean)
  }
  return(list("cov"=cov.matrices,"invcov"=invcov.matrices,"mu"=mu.matrix))
}

computeMoments2<-function(data){
  cov.matrices<-array(0,dim=c(256,256,5))
  invcov.matrices<-array(0,dim=c(256,256,5))
  mu.matrix<-array(0,dim=c(5,256))
  
  for(i in c(1:5)){
    print(i)
    mcd.obj <- covPCAproj(data[which(data$V258 == i), c(1:256)],control=list("CalcMethod" = "lincomb"))
    cov.matrices[,,i]<-mcd.obj$cov
    invcov.matrices[,,i]<-matrix.inverse(cov.matrices[,,i])
    mu.matrix[i,]<-mcd.obj$center
  }
  return(list("cov"=cov.matrices,"invcov"=invcov.matrices,"mu"=mu.matrix))
}

computeMoments3<-function(data){
  cov.matrices<-array(0,dim=c(256,256,5))
  invcov.matrices<-array(0,dim=c(256,256,5))
  mu.matrix<-array(0,dim=c(5,256))
  
  for(i in c(1:5)){
    print(i)
    mcd.obj <- cov.rob(data[which(data$V258 == i), c(1:256)],method='mcd',nsamp='sample')
    cov.matrices[,,i]<-mcd.obj$cov
    invcov.matrices[,,i]<-matrix.inverse(cov.matrices[,,i])
    mu.matrix[i,]<-mcd.obj$center
  }
  return(list("mcdcov"=cov.matrices,"invmcdcov"=invcov.matrices,"mu"=mu.matrix))
}

computeCPCcov<-function(cpc.obj){
  invcov.matrices<-array(0,dim=c(256,256,5))
  U <- cpc.obj$CPC
  Sigma <- cpc.obj$D
  for(i in c(1:5)){
    invcov.matrices[,,i]<-U%*%diag(1/Sigma[,i])%*%t(U)
  }
  invcov.matrices
}

iter <- 5
num.model <- 4

MSE.matrix <- matrix(data=NA,ncol=num.model,nrow=iter)
MSPE.matrix <- matrix(data=NA,ncol=num.model,nrow=iter)

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
  train.data.unique<-phoneme[unique(train.data.ind),]
  
#   obj.pca<-prcomp(train.data[,1:256])
#   mu<-apply(train.data[,1:256],2,mean)
#   sdev<-apply(train.data[,1:256],2,sd)
#   train.data[,1:256]<-train.data[,1:256]-t(matrix(data=rep(mu,times=nrow(train.data)),nrow=p))
#   train.data.unique[,1:256]<-train.data.unique[,1:256]-t(matrix(data=rep(mu,times=nrow(train.data.unique)),nrow=p))
#   test.data[,1:256]<-test.data[,1:256]-t(matrix(data=rep(mu,times=nrow(test.data)),nrow=p))
#   
#   train.data[,1:256]<-train.data[,1:256]/t(matrix(data=rep(sdev,times=nrow(train.data)),nrow=p))
#   train.data.unique[,1:256]<-train.data.unique[,1:256]/t(matrix(data=rep(sdev,times=nrow(train.data.unique)),nrow=p))
#   test.data[,1:256]<-test.data[,1:256]/t(matrix(data=rep(sdev,times=nrow(test.data)),nrow=p))
  
#   U<-obj.pca$rotation
#   diag.vec<-(obj.pca$sdev)^2
#   whiten<-U%*%diag(1/(diag.vec))%*%t(U)
#   train.data[,1:256]<-as.matrix(train.data[,1:256])%*%whiten
#   train.data.unique[,1:256]<-as.matrix(train.data.unique[,1:256])%*%whiten
#   test.data[,1:256]<-as.matrix(test.data[,1:256])%*%whiten
  
  lda.fit <- lda(x=train.data[,c(1:256)], grouping=train.data[,257])
  pred.lda.tr <- predict(lda.fit, newdata=train.data[,c(1:256)])
  pred.lda.te <- predict(lda.fit, newdata=test.data[,c(1:256)])
  lda.pred.train <- pred.lda.tr$class
  lda.pred.test <- pred.lda.te$class
  MSE.matrix[i,1] <- mean(ifelse(lda.pred.train == train.data[,257], yes=0, no=1))
  MSPE.matrix[i,1] <- mean(ifelse(lda.pred.test == test.data[,257], yes=0, no=1))
  
  param1 <- computeMoments1(train.data)
  pk<-t(as.data.frame(table(train.data$V258))[,2])
  pk<-as.numeric(pk/sum(pk))
  pred.tr<-predict.qda(train.data[,c(1:256)],param1$mu,param1$invcov,pk)
  MSE.matrix[i,2]<-mean(ifelse(pred.tr == train.data$V258, yes=0, no=1))
  pred.te<-predict.qda(test.data[,c(1:256)],param1$mu,param1$invcov,pk)
  MSPE.matrix[i,2]<-mean(ifelse(pred.te == test.data$V258, yes=0, no=1))
  
  param3 <- computeMoments3(train.data.unique)
  pred.tr<-predict.qda(train.data[,c(1:256)],param3$mu,param3$invmcdcov,pk)
  MSE.matrix[i,3]<-mean(ifelse(pred.tr == train.data$V258, yes=0, no=1))
  pred.te<-predict.qda(test.data[,c(1:256)],param3$mu,param3$invmcdcov,pk)
  MSPE.matrix[i,3]<-mean(ifelse(pred.te == test.data$V258, yes=0, no=1))
  

  mod.CPC <- cpc(param3$mcdcov)
  invCPC.cov <-computeCPCcov(mod.CPC)
  pred.tr<-predict.qda(train.data[,c(1:256)],param3$mu,invCPC.cov,pk)
  MSE.matrix[i,4]<-mean(ifelse(pred.tr == train.data$V258, yes=0, no=1))
  pred.te<-predict.qda(test.data[,c(1:256)],param3$mu,invCPC.cov,pk)
  MSPE.matrix[i,4]<-mean(ifelse(pred.te == test.data$V258, yes=0, no=1))

  
#   param2 <- computeMoments2(train.data.unique)
#   pk<-t(as.data.frame(table(train.data.unique$V258))[,2])
#   pk<-as.numeric(pk/sum(pk))
#   pred.tr<-predict.qda(train.data[,c(1:256)],param2$mu,param2$invcov,pk)
#   MSE.matrix[i,4]<-mean(ifelse(pred.tr == train.data$V258, yes=0, no=1))
#   pred.te<-predict.qda(test.data[,c(1:256)],param2$mu,param2$invcov,pk)
#   MSPE.matrix[i,4]<-mean(ifelse(pred.te == test.data$V258, yes=0, no=1))
#   print(MSE.matrix[i,])
#   print(MSPE.matrix[i,])

  print(MSE.matrix[i,])
  print(MSPE.matrix[i,])
}                                                                                                     

test_err_frame <- as.data.frame(MSPE.matrix)

names(test_err_frame)<- c("LDA","QDA","CPC","RobustPCA","MCD","MCDwCPC","NoisyLogistic","L2Logistic")
# c("LDA","QDA","Mul.Logit","LASSO","FullTree","PrunedTree.1se","PrunedTree.min","R.F.")

rowMins <- apply(MSPE.matrix,1,min)
minMatrix <- matrix(data=rep(rowMins,times=6),nrow=iter)
ratio_err_frame <- as.data.frame(sqrt(MSPE.matrix/minMatrix))

names(ratio_err_frame)<-c("LDA","QDA","CPC","RobustPCA","MCD","MCDwCPC","NoisyLogistic","L2Logistic")

win.graph(h=7, w=10, pointsize=12)
boxplot(test_err_frame,main="test-MSPEs vs Models",ylab="root-MSPE")

win.graph(h=7, w=10, pointsize=12)
boxplot(ratio_err_frame,main="root ratios of MSPE to minMSPE vs Models",ylab="ratio")