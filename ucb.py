# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 01:21:41 2019

@author: Abhimanyu_Dwivedi
"""

#importing libraries
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

#importing datasets
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing UCB
import math
N = 10000
d = 10
ads_selected = []
num_of_selections = [0] * d
sum_of_rewards = [0] * d 
total_reward = 0
for i in range(0, N):
    ad = 0
    max_upper_bound = 0
    for j in range(0, d):
        if (num_of_selections[j] > 0) :
            avg_reward = sum_of_rewards[j]/num_of_selections[j]
            delta_i = math.sqrt(3/2 * math.log(i + 1) / num_of_selections[j])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = j
    ads_selected.append(ad)
    num_of_selections[ad] += 1
    reward = dataset.values[i, ad]
    sum_of_rewards[ad] += reward 
    total_reward += reward



plt.hist(ads_selected)
plt.title("ads_selections")
plt.xlabel('Ads')
plt.ylabel('Clicks')
plt.show()