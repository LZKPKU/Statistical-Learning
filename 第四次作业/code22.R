library(splines)
n=100
X = runif(n,0,1)
X = sort(X)
eps = rnorm(n,0,1)
f = function(x){sin(12*(x+0.2))/(x+0.2)}
Y = f(X)+eps

#calculate pointwise
cal_ptv = function(df){
  eye = diag(rep(1,n))
  S_lambda = matrix(0,n,n)
  for (i in 1:n){
    # calculation of $S_lambda$
    tempss = smooth.spline(X,eye[,i],df=df)
    S_lambda[,i] = predict(tempss,X)$y
  }
  return (diag(S_lambda%*%t(S_lambda)))
}

# function to plot 2,3,4
draw<-function(df){
  sd2 = 2*sqrt(cal_ptv(df))
  # fit (x,y)
  ss = smooth.spline(X,Y,df=df)
  plot(X,Y,type='p')
  title(main=paste("df=",df))
  polygon(c(X,rev(X)),
          c(ss$y+sd2,rev(ss$y-sd2)),
          col="lightgoldenrod1",border=NA)
  points(X,Y,lwd=1.5)
  curve(f,col="darkorchid",add=TRUE,lwd=3.5)
  lines(ss,col="forestgreen",lwd=3.5)
} 

m = 25
df = seq(5,15,length=m)
EPE=CV=0
for(i in 1:m){
  epe=cv=0
  ss = smooth.spline(X,Y,df =df[i],cv=TRUE)
  for(j in 1:n){
    epe[j] = (predict(ss,X)$y[j]-f(X[j]))^2
    # (5.26)
    cv[j] = (ss$y[j]-Y[j])^2/
            (1-ss$lev[j])^2
  }
  # (5.25)  sigma_square + MSE
  EPE[i] = sum(epe)/n+1+mean(cal_ptv(df[i]))
  CV[i] = sum(cv)/n
}

par(mfrow=c(2,2))
# plot 1
plot(df,EPE,type='p',col="darkorange2",
     ylim=c(1.0,1.6),pch=20,ylab="EPE  and  CV",
     main= "Cross Validation")
points(df,CV,type='p',col="deepskyblue2",pch=20)
abline(v=which.min(CV))
legend("topright",c("CV","EPE"),
       col=c("darkorange2","deepskyblue2"),
       lty='solid',lwd=3)
# plot 2 3 4
df = c(5,9,15)
for (i in df)
  draw(i)
