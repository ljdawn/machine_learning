 X<-c(159, 280, 101, 212, 224, 379, 179, 264, 222, 362, 168, 250, 149, 260, 485, 170)
 > t.test(X)
"""
	One Sample t-test

data:  X
t = 9.7847, df = 15, p-value = 6.649e-08
alternative hypothesis: true mean is not equal to 0
95 percent confidence interval:
 188.8927 294.1073
sample estimates:
mean of x 
    241.5 
    """
X<-c(78.1,72.4,76.2,74.3,77.4,78.4,76.0,75.5,76.7,77.3)
Y<-c(79.1,81.0,77.3,79.1,80.0,79.1,79.1,77.3,80.2,82.1)
t.test(X, Y, var.equal=TRUE, alternative = "less")

X<-c(78.1,72.4,76.2,74.3,77.4,78.4,76.0,75.5,76.7,77.3)
Y<-c(79.1,81.0,77.3,79.1,80.0,79.1,79.1,77.3,80.2,82.1)
t.test(X-Y, alternative = "less")

X<-c(136,144,143,157,137,159,135,158,147,165)
Y<-c(158,142,159,150,156,152,140,149,148,155)
var.test(X,Y)

binom.test(445,500,p=0.85)

X<-c(210, 312, 170, 85, 223)
chisq.test(X)
