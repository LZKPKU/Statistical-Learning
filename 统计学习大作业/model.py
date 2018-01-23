import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import ensemble
import os

trainFile = "E:/My University/My Course/2017-2018第一学期/统计学习/大作业/train.csv"
testFile = "E:/My University/My Course/2017-2018第一学期/统计学习/大作业/test.csv"

os.chdir(os.path.dirname(trainFile))
train = pd.read_csv(os.path.basename(trainFile))
test = pd.read_csv(os.path.basename(testFile))

train_y = train.target
yhatid = test["id"]
del train["target"],train["id"],test["id"]


def My_predict(model,test,name):
    # prediction function
    yhat = model.predict(test)
    yhat[yhat <= 0] = 0
    yhat = pd.DataFrame(yhat)
    yhat.columns = ["target"]
    yhat["id"] = yhatid
    yhat = yhat.reindex_axis(sorted(yhat.columns), axis=1)
    yhat.to_csv(name, header=True, index=False)



OLS = sklearn.linear_model.LinearRegression().fit(train,train_y)
My_predict(OLS,test,"OLS.csv")

Ridge = sklearn.linear_model.RidgeCV().fit(train,train_y)
My_predict(Ridge,test,"Ridge.csv")

Lasso = sklearn.linear_model.LassoCV().fit(train,train_y)
My_predict(Lasso,test,"Lasso.csv")

LR = sklearn.linear_model.LogisticRegressionCV().fit(train,train_y)
My_predict(LR,test,"LR.csv")

