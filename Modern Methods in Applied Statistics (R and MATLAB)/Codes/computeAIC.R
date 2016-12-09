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

p<-ncol(phoneme)-1
numClasses <- 5


det.pop.cov <- det(pop.cov)
det.pop <- rep(det.pop.cov,times=numClasses)

param1 <- computeMoments1(phoneme)

ni<-t(as.data.frame(table(phoneme$V258))[,2])
ni<-as.numeric(ni)-1

pop.cov <- matrix(data=rep(0,times=p*p),ncol=p)
for(i in c(1:numClasses)){
  pop.cov <- pop.cov + ni[i]*class.cov[,,i]
}
pop.cov <- pop.cov/(nrow(phoneme)-numClasses)

log.det.sum<-0
for(i in c(1:numClasses)){
  log.det.sum <- log.det.sum + ni[i]*log(det(class.cov[,,i]))
}
chi2 <- ((nrow(phoneme)-numClasses)*log(det(pop.cov))-log.det.sum)/(1+(1/(3*(numClasses-1)))*(sum(1/ni)-1/(nrow(phoneme)-numClasses)))


computeLoglik<-function(cov.mat,numClasses){
  temp.vec <- rep(0,times=numClasses)
  for(i in c(1:numClasses)){
    temp.vec[i] <- det(cov.mat[,,i])
  }
  temp.vec
}

param2 <- computeMoments2(phoneme)
det.all<-computeLoglik(param1$cov,numClasses)
mod.CPC <- cpc(param1$cov)

cpc.cov.mat<-array(0,dim=c(256,256,numClasses))

U<--mod.CPC$CPC
D<-mod.CPC$D
for(i in c(1:numClasses)){
  cpc.cov.mat[,,i]<-U%*%diag(D[,i])%*%t(U)
}
det.cpc<-computeLoglik(cpc.cov.mat,numClasses)
det.rpca<-computeLoglik(param2$mcdcov,numClasses)
dfs <- c(32896,33920,164480)
computeAIC<-function(det.up,det.low,ni,df){
  sum(ni*log(det.up/det.low))+df
}

