library(MASS)
library(splines)
setwd("E:\\My University\\My Course\\2017-2018第一学期\\统计学习\\第四次作业")
data = read.csv("data.txt")
n = 4500
X = as.matrix(data[1:n,2:257])
Y = data[1:n,258]

fold=matrix(0,nrow=450,ncol=10)
for(i in 1:10)
fold[,i] = ((i-1)*450+1):(i*450)

ss = function(x,df){
  f=1:256
  basis = ns(f,
             knots=seq(8,248,length.out = df)[2:df],
             Boundary.knots = c(8,248)
             )
  H = matrix(0,nrow=n,ncol=df+1)
  for(i in 1:n){
    mo = lm(x[i,]~basis)
    H[i,] = mo$coefficients
  }
  return(H)
}

dfarray=c(8,16,24,32,40)
cv=vector()

for(df in dfarray){
  H = ss(X,df)
  count=0
  for(i in 1:10){
  s = fold[,i]
  model = qda(H[-s,],Y[-s])
  for(j in s){
      if(predict(model,H[j,])$class==Y[j]){
      count=count+1
      }
    }
  }
  cv = c(cv,count/n)
}
cv
