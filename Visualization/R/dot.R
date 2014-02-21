library(ggplot2)
Agw = read.csv("Agent_w_2_pg", header = TRUE, sep = "\t")
Sow = read.csv("Soft_w_2_pg", header = TRUE, sep = "\t")
p0 <- ggplot(data=Agw,aes(x = days_0, y = times_0), colour = factor(days))
p1 <- ggplot(data=Sow,aes(x = days_1, y = time_1), colour = factor(days))
p_0 <- p0 + geom_point() + scale_y_log10() + labs(title = "Agent In") + xlab("days pg 2 kb") + ylab("time counts(in log)")
p_0
p_1 <- p1 + geom_point() + scale_y_log10() + labs(title = "Robot In") + xlab("days pg 2 kb") + ylab("time counts(in log)")
p_1