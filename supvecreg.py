# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:52:34 2019

@author: Abhimanyu_Dwivedi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data extraction
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc1 = StandardScaler()
X = sc.fit_transform(X)
y = sc1.fit_transform(y)

#fitting svr to the dataset
from sklearn.svm import SVR
reg = SVR(kernel = 'rbf')
reg.fit(X, y)

#predict
y_pred = sc1.inverse_transform(reg.predict(sc.transform(np.array([[6.5]]))))

#visualize (X-grid for smooth finish)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()