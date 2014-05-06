from math import *
from matplotlib.pylab import *
import matplotlib
import json

s = json.load(open("style/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)


n=10000
rho=0.79 #correlation
#Means
m1 = 10
m2 = 20
#Standard deviations
s1 = 1
s2 = 2
#Initialize vectors
x=zeros(n, float)
y=zeros(n, float)
sd=sqrt(1-rho**2)
# the core of the method: sample recursively from two normal distributions
# Tthe mean for the current sample, is updated at each step.
for i in range(1,n):
  x[i] = normal(m1+rho*(y[i-1]-m2)/s2,s1*sd)
  y[i] = normal(m2+rho*(x[i]-m1)/s1,s2*sd)

scatter(x,y,marker='d',c='r')

show()