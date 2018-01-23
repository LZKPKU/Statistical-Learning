setwd("E:/My University/My Course/2017-2018第一学期/统计学习/大作业")
train = read.csv("train.csv",header = TRUE)
test = read.csv("test.csv",header = TRUE)


train = train[,-1]
test = test[,-1]

OLS = lm(target~.,data=train)

summary(OLS)
