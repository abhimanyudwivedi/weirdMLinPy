# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 00:24:07 2019

@author: Abhimanyu_Dwivedi
"""

#import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#using the dendrogram
import scipy.cluster.hierarchy as sch
dg = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('DENDROGRAM')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

#fitting hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualizing
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], c = 'red', label = '1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], c = 'blue', label = '2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], c = 'green', label = '3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], c = 'cyan', label = '4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], c = 'magenta', label = '5')
plt.title('HC')
plt.xlabel('Income')
plt.ylabel('Spending')
#plt.legend()
plt.show()