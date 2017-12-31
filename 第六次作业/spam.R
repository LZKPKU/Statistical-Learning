library(mgcv)
library(tree)
library(prim)
library(earth)
library(gbm)
# Part I GAM
setwd("E:/My University/My Course/2017-2018第一学期/统计学习/第六次作业")
data = read.csv("spam.csv",header = FALSE)
testset = array()
trainset = array()
n = nrow(data)
set = 1:n
testset = set[data["V59"]==1]
trainset = set[data["V59"]==0]

trainx = data[,-(58:59)]
trainy = data[,58]
trainx = log(trainx+0.1)

testy = trainy[testset]
trainy = trainy[trainset]
testx = trainx[testset,]
trainx = trainx[trainset,]

V <- paste(paste("s(V",1:57,",k=4)",sep=""),collapse = '+')
form <- paste("trainy~",V,sep = "")
form = formula(form)
tic = Sys.time()
GAM = gam(formula = form,data = as.data.frame(trainx,trainy),family = "binomial")
toc = Sys.time()

GAMpred = predict(GAM,newdata = testx,type="response")
GAMhat = ifelse(GAMpred>=0.5,1,0)

accGAM = sum(as.numeric((GAMhat == testy)))/length(testset)
1-accGAM
toc-tic

# Part II Logistic Regression
tic = Sys.time()
LR = glm(trainy~.,data = data.frame(trainx,trainy),family="binomial")
toc = Sys.time()
LRpred = predict(LR,newdata = data.frame(testx),type="response")
LRhat = ifelse(LRpred>=0.5,1,0)

accLR = sum(as.numeric((LRhat == testy)))/length(testset)
1-accLR
toc-tic
# Part III Classification Tree
trainyy = as.factor(trainy)
testyy = as.factor(testy)
Tree = tree(trainyy~.,data = data.frame(trainx,trainyy),mindev=0)
Treecv = cv.tree(Tree,FUN=prune.misclass,K=10)
size = sort(Treecv$size)[-1]
accTree = (max(Treecv$dev))/3065
for (i in size){
  prune = prune.misclass(Tree,best=i)
  Treepred = predict(prune,newdata = testx,type="class")
  accTree = c(accTree,sum(as.numeric((Treepred != testyy)))/length(testset))
}
plot(sort(Treecv$size) ,accTree,type="b",col="green",ylim=c(0,0.5))
lines(sort(Treecv$size),sort(Treecv$dev/3065,decreasing=TRUE),type="b",col="orange")
i = which(accTree==min(accTree))
1-accTree[i]


# Part IV MARS
tic = Sys.time()
MARS = earth::earth(trainy~.,data = data.frame(trainx,trainy),degree=2)
toc = Sys.time()
MARSpred = predict(MARS,newdata = testx,type="class")
accMARS = sum(as.numeric((MARSpred == testy)))/length(testset)
1-accMARS
toc-tic

# Part V Gradient Boosting Method
tic = Sys.time()
GBM = gbm(trainy~.,data=data.frame(trainx,trainy),n.trees=3000,distribution="bernoulli",
          interaction.depth=5,shrinkage = 0.01, cv.folds=5)
toc = Sys.time()
best.itr = gbm.perf(GBM,method="cv")
best.itr
GBMpred = predict(GBM,newdata = testx,best.itr)
GBMpred = ifelse(GBMpred>=0.5,1,0)
accGBM = sum(as.numeric((GBMpred == testy)))/length(testset)
1-accGBM
toc-tic
