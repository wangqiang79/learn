#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/9 14:04
# @Author  : liuyb
# @Site    : 
# @File    : test.py
# @Software: PyCharm
# @Description:

import matplotlib.pyplot as plt
import seaborn as sns
import slots

# Test multiple strategies for the same bandit probabilities
probs = [0.4, 0.9, 0.8]

strategies = [{'strategy': 'eps_greedy', 'regret': [],
               'label': '$\epsilon$-greedy ($\epsilon$=0.1)'},
              {'strategy': 'softmax', 'regret': [],
               'label': 'Softmax ($T$=0.1)'},
              {'strategy': 'ucb', 'regret': [],
               'label': 'UCB1'},
              {'strategy': 'bayesian', 'regret': [],
               'label': 'Bayesian bandit'},
              ]

for s in strategies:
    s['mab'] = slots.MAB(probs=probs, live=False)

# Run trials and calculate the regret after each trial
for t in range(10000):
    for s in strategies:
        s['mab']._run(s['strategy'])
        s['regret'].append(s['mab'].regret())

# Pretty plotting
sns.set_style('whitegrid')
sns.set_context('poster')

plt.figure(figsize=(15, 4))

for s in strategies:
    plt.plot(s['regret'], label=s['label'])

plt.legend()
plt.xlabel('Trials')
plt.ylabel('Regret')
plt.title('Multi-armed bandit strategy performance (slots)')
plt.ylim(0, 0.2)

plt.show()
