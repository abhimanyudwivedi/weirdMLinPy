# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 21:14:14 2019

@author: Abhimanyu_Dwivedi
"""
#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data loading
dataset = pd.read_csv('Position_Salaries.csv')

#data locating
X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2:3]

#feature scaling it has no effect though, without feature scaling it returns the same value
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc1 = StandardScaler()
X = sc.fit_transform(X)
y = sc1.fit_transform(y)

#decision tree regression
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state = 0)
reg.fit(X, y)

#predict
y_pred = sc1.inverse_transform(reg.predict(sc.transform(np.array([[6.5]]))))

#Plotting with X_grid is import to show the discontinuity, Decision tree regressor takes the average of each set, hence it should show the discontinuity
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Truth or bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()