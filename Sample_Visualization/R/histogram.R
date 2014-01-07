library(ggplot2)
kb_days = read.csv("closed_kb_days_101006_140107", header = TRUE, sep = "\t")
kb_days_f1 <- data.frame(kb_days[kb_days != 0])
colnames(kb_days_f1)[1] <- "days"
kb_days_f2 = data.frame(kb_days_f1[kb_days_f1 != 1])
colnames(kb_days_f2)[1] <- "days"
p0 <- ggplot(data=kb_days,aes(x=days), colour = factor(cyl))
p1 <- ggplot(data=kb_days_f1,aes(x=days), colour = factor(cyl))
p2 <- ggplot(data=kb_days_f2,aes(x=days), colour = factor(cyl))
p_0 <-p0 + labs(title = "kb_days_histogram") + geom_histogram(binwidth = 1) +
xlab("kb days(binwidth = a week)") + ylab("day counts")
p_0
p_2 <-p2 + labs(title = "kb_days_histogram") + geom_histogram(binwidth = 7) +
  xlab("kb days(binwidth = a week)") + ylab("day counts") + scale_x_continuous(limits=c(0, 365))
p_2