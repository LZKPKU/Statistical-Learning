# Parameter
p = 10;n = 10000;ntest = 10000;M=3000;

# Training Process
X = matrix(nrow=n,ncol=p)
y = vector(length=n)
# generating training data
for (i in 1:p)
  X[,i] = rnorm(n,0,1)
for (i in 1:n)
  y[i] = sum(X[i,]^2)
y[y<=9.34] = -1
y[y>9.34]=1

# save parameters of each classifier
f = vector(length=M)
x = vector(length=M)
s = vector(length=M)
alpha = vector(length=M)

# save weights
w = vector(length=n)
w[1:n] = rep(1/n,1)

# construct data structure
Sorted_X = matrix(nrow=n,ncol=p)
Order = matrix(nrow=n,ncol=p)
Y = matrix(nrow=n,ncol=p)
W = matrix(nrow=n,ncol=p)

for (j in 1:p)
{ #Sorted_X[i,j] stores the i-th smallest value of jth row.
  #X[i,j] belongs to y[Order[i,j]]
  Sorted_X[,j] = sort(X[,j])
  Order[,j] = order(X[,j])
  Y[1:n,j] = y[Order[1:n,j]]
}


# training enumerate each stump
# L_left means < -> 1
# L_right means < -> -1 

L_left = matrix(nrow=n,ncol=p) 
L_right = matrix(nrow=n,ncol=p)

for(t in 1:M){  
  #print(t)
  W[1:n,1:p] = w[Order[1:n,1:p]]
  # dynamic programming
  for(j in 1:p){
    L_left[1,j] = 2+2*sum(W[,j]*Y[,j])
    for(i in 2:n)
      L_left[i,j]=L_left[i-1,j]-Y[i-1,j]*W[i-1,j]
    
    L_right = 4 - L_left
    leftmin = min(L_left)
    rightmin = min(L_right)
  }
  # find min error and its index, save the parameters
  if(leftmin<rightmin){
    s[t] = 1
    r = which(L_left == leftmin, arr.ind = TRUE)[1,1]
    c = which(L_left == leftmin, arr.ind = TRUE)[1,2]
    f[t] = c
    x[t] = Sorted_X[r,c]
    err = leftmin/4
  }
  else{
    s[t] = -1
    r = which(L_right == rightmin, arr.ind = TRUE)[1,1]
    c = which(L_right == rightmin, arr.ind = TRUE)[1,2]
    f[t] = c
    x[t] = Sorted_X[r,c]
    err = rightmin/4
  }
  # calculate alpha
  alpha[t] = log((1-err)/err)
  # update weights
  for(i in 1:n){
    if(s[t] == 1){
      if(X[i,f[t]]<x[t]){
        if(y[i] == -1){
          w[i]=w[i]*exp(alpha[t])
        }
      }
      else{
        if(y[i] == 1){
          w[i]=w[i]*exp(alpha[t])
        }
      }
    }
    else{
      if(X[i,f[t]]<x[t]){
        if(y[i] == 1){
          w[i]=w[i]*exp(alpha[t])
        }
      }
      else{
        if(y[i] == -1){
          w[i]=w[i]*exp(alpha[t])
        }
      }
    }
  }
  # normalization
  w = w/sum(w)
}
# f, s, alpha, x save the model

# Testing Process
#generating test data
Xtest = matrix(nrow=ntest,ncol=p)
ytest = vector(length=ntest)
for (i in 1:p)
  Xtest[,i] = rnorm(n,0,1)
for (i in 1:ntest)
  ytest[i] = sum(Xtest[i,]^2)
ytest[ytest<=9.34] = -1
ytest[ytest>9.34] = 1

# prediction function
Adaboost_predict<-function(Xtest,f,s,alpha,x){
 
  # number of weak classifiers
  M = length(x)
  ntest = nrow(Xtest)
  y_pred = vector(length=ntest)
  for(t in 1:M){
    if (s[t] == 1){
      for (i in 1:ntest){
        if(Xtest[i,f[t]]<x[t]){
          y_pred[i] = y_pred[i] + alpha[t]*1}
        else{
          y_pred[i] = y_pred[i] + alpha[t]*(-1)}
      }
    }
    else{
      for (i in 1:ntest){
        if(Xtest[i,f[t]]<x[t]){
          y_pred[i] = y_pred[i] + alpha[t]*(-1)}
        else{
          y_pred[i] = y_pred[i] + alpha[t]*1}
      }
    }
  }
  y_pred[y_pred<=0] = -1
  y_pred[y_pred>0] = 1
  return(y_pred)
}
# accuracy on training set
y_hat = Adaboost_predict(X,f,s,alpha,x)
acc_train = sum(y_hat==y)/n
# accuracy on testing set
y_pred = Adaboost_predict(Xtest,f,s,alpha,x)
acc_test = sum(y_pred==ytest)/ntest

acc_train
acc_test