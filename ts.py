# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:56:33 2019

@author: Abhimanyu_Dwivedi
"""

#importing libraries 3 months late
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

#importing datasets
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing TS
N = 10000
d = 10
ads_selected = []
num_rewards_won = [0] * d
num_rewards_lost = [0] * d
total_reward = 0
for i in range(0, N):
    ad = 0
    max_random = 0
    for j in range(0, d):
        random_beta = random.betavariate(num_rewards_won[j] + 1, num_rewards_lost[j] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = j
    ads_selected.append(ad)
    reward = dataset.values[i, ad]
    if reward == 1:
        num_rewards_won[ad] += 1
    else:
        num_rewards_lost[ad] += 1
    total_reward += reward



plt.hist(ads_selected)
plt.title("ads_selections")
plt.xlabel('Ads')
plt.ylabel('Clicks')
plt.show()