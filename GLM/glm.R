mydata <- read.table("../testing/data_glm")
my_test <- glm(mydata$V6~mydata$V1+mydata$V2+mydata$V3+mydata$V4+mydata$V5+1, data=mydata, , family = poisson())
y_pre <- predict(my_test, mydata)
print (y_pre)