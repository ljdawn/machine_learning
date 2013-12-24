mydata <- read.table("../testing/data_glm")
my_test <- glm(mydata$V6~mydata$V1+mydata$V2+mydata$V3+mydata$V4+mydata$V5, data=mydata,family = binomial("logit"))
y_pre <- predict(my_test, data.frame(mydata$V1+mydata$V2+mydata$V3+mydata$V4+mydata$V5))
print (exp(y_pre)/(1+exp(y_pre)))