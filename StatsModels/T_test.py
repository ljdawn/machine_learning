import statsmodels.api as sm

tl = [159, 280, 101, 212, 224, 379, 179, 264, 222, 362, 168, 250, 149, 260, 485, 170]
result = sm.stats.DescrStatsW(tl)

#So what? where my p-value

X =[78.1,72.4,76.2,74.3,77.4,78.4,76.0,75.5,76.7,77.3]
Y =[79.1,81.0,77.3,79.1,80.0,79.1,79.1,77.3,80.2,82.1]

result = sm.stats.CompareMeans(X, Y)