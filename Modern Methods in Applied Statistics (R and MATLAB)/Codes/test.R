require(plyr)
require(abind)
library(cpca)

setwd('d:/STAT852/project2')

data(iris)

C <- daply(iris, "Species", function(x) cov(x[, -ncol(x)]))
C <- aperm(C, c(2, 3, 1))

mod1 <- cpc(C)