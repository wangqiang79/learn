#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/26 15:24
# @Site    :
# @File    : __init__.py.py
# @Software: PyCharm
# @Description: Dyna框架的决策类

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # 动作列表
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 回报的衰减值
        self.epsilon = e_greedy  # ε-贪心算法中的ε值
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # q表

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # 动作选择
        if np.random.uniform() < self.epsilon:
            # 当随机值小于ε时,选择q表中当前状态下行为值函数最大的值
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # 因为有些行为的行为值函数相同,随意现在会随机选择一个行为
            action = state_action.idxmax()
        else:
            # 当随机值大于ε时,会随机选择行为
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # 当下一个状态不是终止状态时,使用Q-Learning更新公式进行q_target值的更新
        else:
            q_target = r  # 当下一个状态为终止状态时
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # 更新Q表中当前状态下当前行为的值

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 当当前状态不在Q表中时,将当前状态加入Q表
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def clear_q_table(self):
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


class EnvModel:
    """与DQN中的内存缓冲区类似，可以在此存储过去的经历,该模型可以精确地生成下一个状态和奖励信号"""

    def __init__(self, actions):
        # 最简单的情况是考虑模型是一个拥有所有过去的转换信息的存储器
        self.actions = actions
        self.database = pd.DataFrame(columns=actions, dtype=np.object)

    def store_transition(self, s, a, r, s_):
        if s not in self.database.index:
            self.database = self.database.append(
                pd.Series(
                    [None] * len(self.actions),
                    index=self.database.columns,
                    name=s,
                ))
        self.database.set_value(s, a, (r, s_))

    def sample_s_a(self):
        s = np.random.choice(self.database.index)
        a = np.random.choice(self.database.ix[s].dropna().index)  # 过滤掉None值
        return s, a

    def get_r_s_(self, s, a):
        r, s_ = self.database.ix[s, a]
        return r, s_

    def clear_database(self):
        self.database = pd.DataFrame(columns=self.actions, dtype=np.object)
