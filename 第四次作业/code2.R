# import library
library(glm2)
library(splines)
library(ggplot2)
# data input and preprocessing
data = read.csv(file.choose())
aa = data[data['g']=="aa",]
ao = data[data['g']=="ao",]
aa = subset(aa,select=-c(1,259))
ao = subset(ao,select=-c(1,259))
Phoneme = rbind(aa,ao)
# Normal logistic regression
logit.fit = glm(g~.,family = binomial(link='logit'),data=Phoneme)
# Sharp coefficients
ori_beta = logit.fit$coefficients[-1]
#plot(logit.fit$coefficients,type='l',ylim=c(-0.4,0.4))
# regard beta as function of frequency
f = 1:256
# splines 
model = ns(f,df=12)
H = t(model)
# transform data to X*
X= t(Phoneme[,1:256])
X=H%*%X
X = t(X)
Y= cbind(X,Phoneme['g'])
# New Logistic Regression
logit.fitns = glm(g~.,family=binomial(link='logit'),data=Y)
theta = logit.fitns$coefficients[-1]
# get beta 
beta = theta%*%H
beta = beta[1,]
# Smooth coefficients
#plot(beta[1,1:256],type='l',ylim=c(-0.4,0.4))

p = ggplot()
p+geom_point(aes(x=f,y=ori_beta),color="gray")+geom_line(aes(x=f,y=ori_beta,'qsec'),color="gray")+
geom_point(aes(x=f,y=beta),color="orange")+geom_line(aes(x=f,y=beta,'qsec'),color="orange")+
  scale_y_continuous("Coefficients")+scale_x_continuous("Frequency")

