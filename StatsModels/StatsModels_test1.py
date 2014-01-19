#coding:utf8
import numpy as np
import statsmodels.api as sm

nsample = 100
x = np.linspace(0,10, 100)
X = sm.add_constant(np.column_stack((x, x**2)))
beta = np.array([1, 0.1, 10])
y = np.dot(X, beta) + np.random.normal(size=nsample)

results = sm.OLS(y, X).fit()
print results.summary()
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 4.234e+06
Date:                Mon, 20 Jan 2014   Prob (F-statistic):          2.30e-240
Time:                        01:52:36   Log-Likelihood:                -143.99
No. Observations:                 100   AIC:                             294.0
Df Residuals:                      97   BIC:                             301.8
Df Model:                           2                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const          1.0733      0.305      3.520      0.001         0.468     1.678
x1             0.0756      0.141      0.537      0.593        -0.204     0.355
x2            10.0057      0.014    733.690      0.000         9.979    10.033
==============================================================================
Omnibus:                        4.400   Durbin-Watson:                   2.271
Prob(Omnibus):                  0.111   Jarque-Bera (JB):                4.019
Skew:                           0.489   Prob(JB):                        0.134
Kurtosis:                       3.091   Cond. No.                         144.
==============================================================================
"""