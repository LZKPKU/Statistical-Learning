# pointwise calculation function
ptv<-function(model,X){
  p = length(model$coefficients)
  res = sum((model$residuals)^2)/(n-p)
  X = cbind(one,X)
  inv = solve(t(X)%*%X)
  var = 0
  V = X%*%(inv*res)%*%t(X)
  return (diag(V))
}
library(splines)
library(ggplot2)
# random initialization
n=50
set.seed(200)
x = runif(n,min=0,max=1)
x = sort(x)
y = rnorm(n,0,1)
one =seq(1,1,length.out=n)
# global linear
model1 = lm(y~x)
Var1 = ptv(model1,x)
# global cubic
X2 = cbind(x,x^2,x^3)
model2 = lm(y~X2)
Var2 = ptv(model2,X2)
# cubic spline
linear_spline = bs(x,degree=3,knots=c(1/3,2/3))
model3 = lm(y~linear_spline)
Var3 = ptv(model3,linear_spline)
# natural cubic spline
natural_spline = ns(x,df=6,knots=seq(0.1,0.9,length.out=6))
model4 = lm(y~natural_spline)
Var4 = ptv(model4,natural_spline)
# plot
result = data.frame(x,Var1,Var2,Var3,Var4)
p = ggplot(data=result)
p+geom_point(aes(x=x,y=Var4,color="Natural Cubic Spline - 6 knots"),size=2)+
geom_line(aes(x=x,y=Var4,color="Natural Cubic Spline - 6 knots"),size=1)+
 geom_point(aes(x=x,y=Var3,color="Cubic Spline - 2knots"),size=2)+
  geom_line(aes(x=x,y=Var3,color="Cubic Spline - 2knots"),size=1)+
 geom_point(aes(x=x,y=Var2,color="Global Cubic Polynomial"),size=2)+
  geom_line(aes(x=x,y=Var2,color="Global Cubic Polynomial"),size=1)+
 geom_point(aes(x=x,y=Var1,color="Global Linear"),size=2)+
  geom_line(aes(x=x,y=Var1,color="Global Linear"),size=1)+
 scale_y_continuous("Pointwise Variance")+scale_x_continuous("X")+
  guides(color=guide_legend(title=NULL))+theme(legend.position = 'top')+
  scale_color_discrete(breaks = c('Global Linear','Global Cubic Polynomial',
                'Cubic Spline - 2knots','Natural Cubic Spline - 6 knots'))+
  theme(panel.grid.major =element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(),axis.line = element_line(colour = "black"))

