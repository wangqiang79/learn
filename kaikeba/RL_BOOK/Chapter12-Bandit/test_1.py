#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/9 14:43
# @Author  : liuyb
# @Site    : 
# @File    : test_1.py
# @Software: PyCharm
# @Description:

import matplotlib.pyplot as plt
import seaborn as sns
from KArmedBandit import Operate

# strategies = [{'strategy': 'eps_greedy', 'regret': [],
#                'label': '$\epsilon$-greedy ($\epsilon$=0.1)'},
#               {'strategy': 'softmax', 'regret': [],
#                'label': 'Softmax ($T$=0.1)'},
#               {'strategy': 'ucb', 'regret': [],
#                'label': 'UCB1'},
#               {'strategy': 'bayesian', 'regret': [],
#                'label': 'Bayesian bandit'},
#               ]
# probs = [0.4, 0.9, 0.8]
#
# for s in strategies:
#     s['mab'] = Operate(probs=probs)
#
# # Run trials and calculate the regret after each trial
# for t in range(10000):
#     for s in strategies:
#         s['mab']._run(s['strategy'])
#         s['regret'].append(s['mab'].regret())
#
# # Pretty plotting
# sns.set_style('whitegrid')
# sns.set_context('poster')
#
# plt.figure(figsize=(15, 4))
#
# for s in strategies:
#     plt.plot(s['regret'], label=s['label'])
#
# plt.legend()
# plt.xlabel('Trials')
# plt.ylabel('Regret')
# plt.title('Multi-armed bandit strategy performance')
# plt.ylim(0, 0.2)
#
# plt.show()

import numpy as np

num_bandits = 10
wins = np.zeros(num_bandits)
pulls = np.zeros(num_bandits)

print(wins)
print(pulls)

p_success_arms = [
            np.random.beta(wins[i] + 1, pulls[i] - wins[i] + 1)
            for i in range(len(wins))
            ]

print(p_success_arms)