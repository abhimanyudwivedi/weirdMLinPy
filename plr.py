# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 03:29:47 2019

@author: Abhimanyu_Dwivedi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

#Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#linear regression for polynomials Haahahahahhahahahahcryinghaahhahhahha
lr2 = LinearRegression()
lr2.fit(X_poly, y)

#visualising linear regression's results
plt.scatter(X, y, color = 'red')
plt.plot(X, lr.predict(X), color = 'green')
plt.title('Truth-or-Bluff')
plt.xlabel('Position_level')
plt.ylabel('Salary')
plt.show()

#visualising Polynomial regression's results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lr2.predict(poly_reg.fit_transform(X_grid)), color = 'green')
plt.title('truth or Bluff')
plt.xlabel('Position_level')
plt.ylabel('Salary')
plt.show()

lr.predict(6.5)

lr2.predict(poly_reg.fit_transform(6.5))