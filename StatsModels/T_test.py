import statsmodels.api as sm

tl = [159, 280, 101, 212, 224, 379, 179, 264, 222, 362, 168, 250, 149, 260, 485, 170]
result = sm.stats.DescrStatsW(tl)

#So what? where my p-value