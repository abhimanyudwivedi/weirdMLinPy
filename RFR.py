 # -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:35:03 2019

@author: Abhimanyu_Dwivedi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#creating the regressor
from sklearn.ensemble import RandomForestRegressor
ref = RandomForestRegressor(n_estimators = 250, random_state = 0)
ref.fit(X, y)

#prediction

y_pred = ref.predict(6.5)

#visualize with high resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, ref.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()