# Parameter
p = 10;n = 2000

# Training Process
X = matrix(nrow=n,ncol=p)
y = vector(length=n)
y_hat = vector(length=n)
err = vector(length=n)
# generating training data
for (i in 1:p)
  X[,i] = rnorm(n,0,1)
for (i in 1:n)
  y[i] = sum(X[i,]^2)

y[y<=9.34] = -1
y[y>9.34]=1

# save Decision stump parameter
f = vector(length=n*p*2)
x = vector(length=n*p*2)
s = vector(length=n*p*2)
alpha = vector(length=n*p*2)

# save weights
w = vector(length=n)
w[1:n] = 1

# Maximum iterate number
M = 400
err0 = err1 = 0

# training enumerate each stump
for(t in 1:M){  
  minerr = Inf
  sumw = sum(w)
  for (i in 1:n){
   for (j in 1:p){
     for(k in 1:2){
       yerr = 0
        if (k==1){
          for(l in 1:n){
            if(X[l,j]<=X[i,j]){
              if(y[l] == -1){
                yerr = yerr + w[l]
              }
            }
            else{
              if(y[l] == 1){
                yerr = yerr + w[l]
              }
            }
          }
        }
       else{
         for(l in 1:n){
           if(X[l,j]<=X[i,j]){
             if(y[l] == 1){
              yerr = yerr + w[l]
             }
           }
           else {
             if(y[l] == -1){
               yerr = yerr + w[l]
             }
           }
         }
       }
       # now we get a yerr
       # min in t iterate, save parameters
       if(yerr<minerr){
         f[t] = j
         x[t] = X[i,j]
         s[t] = k
         minerr = yerr
       }
      }
    }
  }
  # calculate alpha
  err1 = minerr/sumw
  if(abs(err1-err0)<1e-6)
    break;
  err0=err1
  alpha[t] = log((1-err1)/err1)
  # update weights
  for (a in 1:n){
    if(s[t] == 1){
      # y_hat = 1
      if(X[a,f[t]]<=x[t]){
        y_hat[a] = y_hat[a]+alpha[t]*1
        # wrong prediction
        if(y[a]!=1){
          w[a] = w[a]*exp(alpha[t])
        }
      }
      # y_hat = -1
      else{
        y_hat[a] = y_hat[a]+alpha[t]*(-1)
        # wrong prediction
        if(y[a]==1){
          w[a] = w[a]*exp(alpha[t])
        }
      }
    }
    else{
      # y_hat = -1
      if(X[a,f[t]]<=x[t]){
        y_hat[a] = y_hat[a]+alpha[t]*(-1)
        # wrong prediction
        if(y[a]==1){
          w[a] = w[a]*exp(alpha[t])
        }
      }
      # y_hat = 1
      else{
        y_hat[a] = y_hat[a]+alpha[t]*1
        # wrong prediction
        if(y[a]!=1){
          w[a] = w[a]*exp(alpha[t])
        }
      }
    }
  }
}

# prediction on training data
y_hat[y_hat<=0] = -1
y_hat[y_hat>0] = 1
# f, s, alpha, x save the model

# Testing Process
ntest=10000
Xtest = matrix(nrow=ntest,ncol=p)
ytest = vector(length=ntest)
y_pred = vector(length=ntest)
y_pred[1:ntest] = 0

for (i in 1:p)
  Xtest[,i] = rnorm(n,0,1)
for (i in 1:ntest)
  ytest[i] = sum(Xtest[i,]^2)

ytest[ytest<=9.34] = -1
ytest[ytest>9.34]=1



for(t in 1:M){
  if (s[t] == 1){
    for (i in 1:ntest){
      if(Xtest[i,f[t]]<=x[t]){
        y_pred[i] = y_pred[i] + alpha[t]*1}
      else{
        y_pred[i] = y_pred[i] + alpha[t]*(-1)}
    }
  }
  else{
    for (i in 1:ntest){
      if(Xtest[i,f[t]]<=x[t]){
        y_pred[i] = y_pred[i] + alpha[t]*(-1)}
      else{
        y_pred[i] = y_pred[i] + alpha[t]*1}
    }
  }
}

y_pred[y_pred<=0] = -1
y_pred[y_pred>0] = 1
sum(y_pred==ytest)/ntest