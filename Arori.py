# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:49:19 2019

@author: Abhimanyu_Dwivedi
"""

#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

#taining Apriori in dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#visualising
res = list(rules)