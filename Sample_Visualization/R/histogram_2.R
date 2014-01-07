library(ggplot2)
CC = read.csv("cc_2_end_days", header = TRUE, sep = "\t")
p <- ggplot(data=CC,aes(x=times))
p + geom_histogram(binwidth = 1) + labs(title = "cc 2 end") + xlab("days(binwidth = One day)") + ylab("times") + scale_x_continuous(limits=c(0, 1000))