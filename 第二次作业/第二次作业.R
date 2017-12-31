#########################################
#function calculating testerror&stderror#
#########################################
cal_err<-function(y,yhat){
  len = length(y)
  tvar = sum((y-yhat)^2)/len
  terr = sqrt(tvar)
  stderr = sqrt(sum(((y-yhat)^2-tvar)^2)/(len*(len-1)))
  c(tvar,stderr)
}


###########
#read data#
###########
rawdata = read.table(file.choose())
tr = rawdata['train']
rawdata['train']=NULL
Y = rawdata['lpsa']
data = rawdata[,1:8]
data = scale(data,TRUE,TRUE)
train = data[tr$train,]
test = data[!tr$train,]
Xtrain = as.matrix(train[,1:8])
Ytrain = Y[tr$train,]
Xtest = as.matrix(test[,1:8])
Xtest = cbind(1,Xtest)
Ytest = Y[!tr$train,]

################################
#PART I Linear Regression Model#
################################
LRM = lm(Ytrain~Xtrain)
summary(LRM)
#######################################
# claculate yhat test error&std error #
#######################################
LRM_coef = LRM$coefficients
yhat = Xtest%*%LRM_coef
LRM_testerr = cal_err(Ytest,yhat)[1]
LRM_stderr = cal_err(Ytest,yhat)[2]

########################
#PART II Best Selection#
########################
library("MASS")
library("leaps")
library("lars")

BS = lm(Ytrain~Xtrain)
BS_chosen = stepAIC(BS,direction = "backward")



train = data.frame(train)
BS = regsubsets(lpsa~lcavol+lweight+age+lbph+svi+lcp+gleason+pgg45,data=rawdata,method="exhaustive")
summary(BS)


BS = lars(Xtrain,Ytrain,type="stepwise",normalize=TRUE)
BS
BScv=cv.lars(Xtrain,Ytrain,K=10,type='stepwise')
BS = lm(Ytrain~Xtrain[,1]+Xtrain[,2])
BS_coef = coef(BS)
BS_coef
yhat = Xtest[,1:3]%*%BS_coef
BS_testerr = cal_err(Ytest,yhat)[1]
BS_stderr = cal_err(Ytest,yhat)[2]
BS_testerr
BS_stderr






###########################
#PART III Ridge Regression#
###########################
library(glmnet)
RR = glmnet(x = Xtrain,y = Ytrain,family="gaussian",alpha=0)
summary(RR)
plot(RR,xvar = "lambda")

RRcv = cv.glmnet(x=Xtrain,y=Ytrain,family = "gaussian",alpha = 0,nfold=10)
plot(RRcv)

RR_chosen = glmnet(x=Xtrain,y=Ytrain,family="gaussian",alpha=0,lambda=RRcv$lambda.min)

RR_coef = coef(RR_chosen,mode="fraction")

yhat = Xtest%*%RR_coef
RR_testerr = cal_err(Ytest,yhat)[1]
RR_stderr = cal_err(Ytest,yhat)[2]

##########################
#PART IV LASSO Regression#
##########################
lasso = glmnet(x=Xtrain,y=Ytrain,family="gaussian",alpha=1)
plot(lasso,xvar="lambda")
lassocv = cv.glmnet(x=Xtrain,y=Ytrain,family="gaussian",alpha=1)
plot(lassocv)

# 
# lasso_chosen = glmnet(x=Xtrain,y=Ytrain,family="gaussian",alpha=1,lambda = lassocv$lambda.1se)
# lasso_coef = coef(lasso_chosen)
# 
# yhat = Xtest%*%lasso_coef
# lasso_testerr = cal_err(Ytest,yhat)[1]
# lasso_stderr = cal_err(Ytest,yhat)[2]


lasso_chosen = glmnet(x=Xtrain,y=Ytrain,family="gaussian",alpha=1,lambda = lassocv$lambda.min)
lasso_coef = coef(lasso_chosen)

yhat = Xtest%*%lasso_coef
lasso_testerr = cal_err(Ytest,yhat)[1]
lasso_stderr = cal_err(Ytest,yhat)[2]
