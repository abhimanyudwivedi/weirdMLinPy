# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 22:08:42 2019

@author: Abhimanyu_Dwivedi
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])
oe = OneHotEncoder(categorical_features = [3])
X = oe.fit_transform(X).toarray()

#Avoiding the dummy variable
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

#building an optimal solution using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
lr_ols = sm.OLS(endog = y, exog= X_opt).fit()
lr_ols.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
lr_ols = sm.OLS(endog = y, exog= X_opt).fit()
lr_ols.summary()
X_opt = X[:, [0, 1, 3, 5]]
lr_ols = sm.OLS(endog = y, exog= X_opt).fit()
lr_ols.summary()
X_opt = X[:, [0, 3, 5]]
lr_ols = sm.OLS(endog = y, exog= X_opt).fit()
lr_ols.summary()
X_opt = X[:, [0, 3]]
lr_ols = sm.OLS(endog = y, exog= X_opt).fit()
lr_ols.summary()