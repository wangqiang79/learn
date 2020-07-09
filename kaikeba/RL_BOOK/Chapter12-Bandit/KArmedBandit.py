#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/9 14:12
# @Author  : liuyb
# @Site    : 
# @File    : KArmedBandit.py
# @Software: PyCharm
# @Description: 多臂赌博机问题

import numpy as np

class Bandits(object):
    '''
    定义赌博机类
    '''

    def __init__(self, probs, rewards):
        '''
        :param probs: float数组,赌博机获胜的概率数组
        :param rewards: float数组,获胜时的回报值
        '''

        if len(probs) != len(rewards):
            raise Exception('获胜概率数组和回报数组长度不匹配!')
        self.probs = probs
        self.rewards = rewards

    def pull(self, i):
        '''
        从第i号赌博机上返回的奖励
        :param i: int型数据,赌博机的编号
        :return: float or None
        '''

        # 随机一个float数,当小于传入概率时,返回回报值
        if np.random.rand() < self.probs[i]:
            return self.rewards[i]
        else:
            return 0.0

class Algorithm(object):

    def __init__(self, operate):
        self.operate = operate

    '''
    定义算法类
    '''
    def eps_greedy(self, params):
        '''
        运行ε-贪心策略解决多臂赌博机问题
        :param params:
        :return: 赌博机的编号
        '''
        if params and type(params) == dict:
            eps = params.get('epsilon')
        else:
            eps = 0.1

        r = np.random.rand()

        if r < eps:
            return np.random.choice(list(set(range(len(self.operate.wins))) - {np.argmax(self.operate.wins / (self.operate.pulls + 0.1))}))
        else:
            return np.argmax(self.operate.wins / (self.operate.pulls + 0.1))

    def ucb(self, params=None):
        '''
        运行UCB策略,该策略使用的算法为UCB1
        :param params: None, 为保证API的一致性添加,该算法没用到此参数
        :return: 赌博机的编号
        '''
        if True in (self.operate.pulls < self.operate.num_bandits):
            return np.random.choice(range(len(self.operate.pulls)))
        else:
            n_tot = sum(self.operate.pulls)
            rewards = self.operate.wins / (self.operate.pulls + 0.1)
            ubcs = rewards + np.sqrt(2*np.log(n_tot)/self.operate.pulls)

            return np.argmax(ubcs)

class Operate(object):
    '''
    定义操作类
    '''
    def __init__(self, num_bandits=10, probs=None, rewards=None,strategies=['eps_greedy', 'ucb']):
        '''
        :param num_bandits: int, 赌博机数量(默认: 10)
        :param probs: float型数组, 获胜的概率
        :param rewards: float型数组, 获胜后的回报值
        :param strategies: str数组, 传入的策略值
        '''
        self.choices = []

        # 根据传入的赌博机的获胜概率和获胜奖励值创建赌博机对象
        if not probs:
            if not rewards:
                self.bandits = Bandits(probs=[np.random.rand() for idx in range(num_bandits)],
                                       rewards=np.ones(num_bandits))
            else:
                self.bandits = Bandits(probs=[np.random.rand() for idx in range(len(rewards))], rewards=rewards)
                num_bandits = len(rewards)
        else:
            if rewards:
                self.bandits = Bandits(probs=probs, rewards=rewards)
                num_bandits = len(rewards)
            else:
                self.bandits = Bandits(probs=probs, rewards=np.ones(len(probs)))
                num_bandits = len(probs)

        self.num_bandits = num_bandits
        self.wins = np.zeros(num_bandits)
        self.pulls = np.zeros(num_bandits)

        # 赌博机的可选策略,默认为算法类中的全部策略
        self.strategies = strategies
        # 创建算法对象
        self.algorithm = Algorithm(self)

    def run(self, time=100, strategy='eps_greedy', parameters={'epsilon': 0.1}):
        '''
        运行操作time次
        :param time: int, 运行次数
        :param strategy: str, 策略名称,默认为ε-贪心策略
        :param parameters: dict, 策略相关的参数,默认ε=0.1
        :return: None
        '''
        if int(time) < 1:
            raise Exception('运行次数应该大于1!')

        if strategy not in self.strategies:
            raise Exception('传入的策略不支持,请选择策略: {}'.format(', '.join(self.strategies)))

        for n in range(time):
            self._run(strategy, parameters)

    def _run(self, strategy, parameters=None):
        '''
        单次的运行操作中传入的策略
        :param strategy: str, 策略名称
        :param parameters: dict, 策略对应的参数
        :return: None
        '''

        choice = self.run_strategy(strategy, parameters)
        self.choices.append(choice)
        rewards = self.bandits.pull(choice)
        if rewards is None:
            return None
        else:
            self.wins[choice] += rewards
        self.pulls[choice] += 1

    def run_strategy(self, strategy, parameters):
        '''
        运行策略并返回赌博机选择的行为
        :param strategy: str, 策略名称
        :param parameters: dict, 策略对应的参数
        :return: 赌博机的编号
        '''

        return self.algorithm.__getattribute__(strategy)(params=parameters)

    def regret(self):
        '''
        计算期望的后悔值
        预期遗憾 = 最大奖励 - 所收集奖励的总和,例如:  E(target)= T * max_k（mean_k） - sum_（t = 1 - > T）（reward_t）
        :return:
        '''

        return (sum(self.pulls)*np.max(np.nan_to_num(self.wins/(self.pulls + 0.1))) - sum(self.wins)) / (sum(self.pulls) + 0.1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 定义策略字典
    strategies = [{'strategy': 'eps_greedy', 'regret': [],
                   'label': '$\epsilon$-greedy ($\epsilon$=0.1)'},
                  {'strategy': 'ucb', 'regret': [],
                   'label': 'UCB1'}
                  ]
    for s in strategies:
        s['mab'] = Operate()

    for t in range(1000):
        for s in strategies:
            s['mab'].run(strategy=s['strategy'])
            s['regret'].append(s['mab'].regret())

    sns.set_style('whitegrid')
    sns.set_context('poster')

    plt.figure(figsize=(15, 4))
    for s in strategies:
        plt.plot(s['regret'], label=s['label'])

    plt.legend()
    plt.xlabel('Trials')
    plt.ylabel('Regret')
    plt.title('Multi-armed bandit strategy performance')
    # plt.ylim(0, 0.2)

    plt.show()
