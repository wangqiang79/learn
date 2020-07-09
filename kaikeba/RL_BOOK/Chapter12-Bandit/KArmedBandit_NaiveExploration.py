#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 17:09
# @Author  : liuyb
# @Site    : 
# @File    : KArmedBandit_NaiveExploration.py
# @Software: PyCharm
# @Description: Naive Exploration(朴素探索)解决多臂赌博机问题

import matplotlib.pyplot as plt
import numpy as np

# 定义赌博机类
class Bandit:
    # @kArm: 赌博机的臂数
    # @epsilon: ε-贪心算法探索的可能性
    # @initial: 对每个行为评估进行初始化
    # @stepSize: 更新评估的步长大小
    # @sampleAverages: 如果为True，则使用样本平均值来更新估计值而不是常量步长
    def __init__(self, kArm=10, epsilon=0., initial=0., stepSize=0.1, sampleAverages=False, trueReward=0.):
        self.k = kArm
        self.stepSize = stepSize
        self.sampleAverages = sampleAverages
        self.indices = np.arange(self.k)
        self.time = 0
        self.averageReward = 0
        self.trueReward = trueReward

        # 每个行为的真实回报
        self.qTrue = []

        # 每个行为的评估
        self.qEst = np.zeros(self.k)

        # 每个行为选择某号臂的次数
        self.actionCount = []

        self.epsilon = epsilon

        # 使用N（0,1）分布和具有所需初始值的估计来初始化实际奖励
        for i in range(0, self.k):
            self.qTrue.append(np.random.randn() + trueReward)
            self.qEst[i] = initial
            self.actionCount.append(0)

        self.bestAction = np.argmax(self.qTrue)

    # 为当前赌博机获取一个行为,探索或利用
    def getAction(self):
        # 探索
        if self.epsilon > 0:
            if np.random.binomial(1, self.epsilon) == 1:
                np.random.shuffle(self.indices)
                return self.indices[0]

        # 利用
        return np.argmax(self.qEst)

    # 出入一个行为,根据当前行为更新评估结果
    def takeAction(self, action):
        # 生成一个回报 N(真实回报, 1)
        reward = np.random.randn() + self.qTrue[action]
        self.time += 1
        self.averageReward = (self.time - 1.0) / self.time * self.averageReward + reward / self.time
        self.actionCount[action] += 1

        if self.sampleAverages:
            # 使用样本平均值更新估算
            self.qEst[action] += 1.0 / self.actionCount[action] * (reward - self.qEst[action])
        else:
            # 以恒定步长更新估计
            self.qEst[action] += self.stepSize * (reward - self.qEst[action])
        return reward

figureIndex = 0

# 模拟赌博机
def banditSimulation(nBandits, time, bandits):
    bestActionCounts = [np.zeros(time, dtype='float') for _ in range(0, len(bandits))]
    averageRewards = [np.zeros(time, dtype='float') for _ in range(0, len(bandits))]
    for banditInd, bandit in enumerate(bandits):
        for i in range(0, nBandits):
            for t in range(0, time):
                action = bandit[i].getAction()
                reward = bandit[i].takeAction(action)
                averageRewards[banditInd][t] += reward
                if action == bandit[i].bestAction:
                    bestActionCounts[banditInd][t] += 1
        bestActionCounts[banditInd] /= nBandits
        averageRewards[banditInd] /= nBandits
    return bestActionCounts, averageRewards

# 朴素探索
# 当epsilon=0时为贪心算法,当epsilon=0.1时为ε-贪心算法
def naiveExploration(nBandits, time):
    epsilons = [0, 0.1]
    bandits = []
    for epsInd, eps in enumerate(epsilons):
        bandits.append([Bandit(epsilon=eps, sampleAverages=True) for _ in range(0, nBandits)])
    bestActionCounts, averageRewards = banditSimulation(nBandits, time, bandits)

    return epsilons, bestActionCounts, averageRewards

# 绘制结果
def plotResult(epsilons, bestActionCounts, averageRewards):
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    for eps, counts in zip(epsilons, bestActionCounts):
        plt.plot(counts, label='epsilon = ' + str(eps))
    plt.xlabel('Steps')
    plt.ylabel('optimal action')
    plt.legend()
    plt.figure(figureIndex)
    figureIndex += 1
    for eps, rewards in zip(epsilons, averageRewards):
        plt.plot(rewards, label='epsilon = ' + str(eps))
    plt.xlabel('Steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    # 使用朴素探索法解决多臂赌博机问题
    epsilons, bestActionCounts, averageRewards = naiveExploration(10000, 20)
    # 绘制结果
    plotResult(epsilons, bestActionCounts, averageRewards)

