library(ggplot2)
CC = read.csv("cc_2_end_days", header = TRUE, sep = "\t")
KB = read.csv("kb_2_end_days", header = TRUE, sep = "\t")
p <- ggplot(data=CC,aes(x=times))
p1 <- ggplot(data=KB,aes(x=times))
p + geom_histogram(binwidth = 7) + labs(title = "cc 2 end") + xlab("days(binwidth = One week)") + ylab("times") + scale_x_continuous(limits=c(1, 1000))
p1 + geom_histogram(binwidth = 7) + labs(title = "kb 2 end") + xlab("days(binwidth = One week)") + ylab("times") + scale_x_continuous(limits=c(1, 1000))