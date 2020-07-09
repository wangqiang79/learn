#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd

class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 如果状态在当前的Q表中不存在,将当前状态加入Q表中
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # 从均匀分布的[0,1)中随机采样,当小于阈值时采用选择最优行为的方式,当大于阈值选择随机行为的方式,这样人为增加随机性是为了解决陷入局部最优
        if np.random.rand() < self.epsilon:
            # 选择最优行为
            state_action = self.q_table.ix[observation, :]
            # 因为一个状态下最优行为可能会有多个,所以在碰到这种情况时,需要随机选择一个行为进行
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # # 选择随机行为
            action = np.random.choice(self.actions)
        return action

    # 判断下一个状态是否为最优的行为
    def judge(self,observation,action_):
        self.check_state_exist(observation)

        state_action = self.q_table.ix[observation, :]

        max_num = state_action.max()

        idxs = []

        for max_item in range(len(state_action)):
            if state_action[max_item] == max_num:
                idxs.append(max_item)

        if action_ in idxs:
            return True
        return False

    def learn(self, *args):
        pass

# Sarsa(λ)后向资格迹
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 增加新的状态到Q表中
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            # 更新资格迹
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_, ):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            # 下一个状态不是终止状态
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            # 下一个状态是终止状态
            q_target = r
        error = q_target - q_predict

        # 增加访问状态行为对的跟踪数量

        # 方法1:
        self.eligibility_trace.ix[s, a] += 1

        # 方法2:
        # self.eligibility_trace.ix[s, :] *= 0
        # self.eligibility_trace.ix[s, a] = 1

        # Q更新
        self.q_table += self.lr * error * self.eligibility_trace

        # 更新后的衰减资格迹
        self.eligibility_trace *= self.gamma*self.lambda_

# Q(λ)--Watkins's Q(λ)后向资格迹
class QLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(QLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 增加新的状态到Q表中
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            # 更新资格迹
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            # 下一个状态也不是终止状态
            q_target = r + self.gamma * self.q_table.ix[s_, a_].max()
        else:
            # 下一个状态也是终止状态
            q_target = r
        error = q_target - q_predict

        # 判断s'状态下的行为是否为最优行为
        a_flag = self.judge(s,a_)

        # 增加访问状态行为对的跟踪数量

        # 资格迹+1
        self.eligibility_trace.ix[s, a] += 1

        # 方法2:
        # self.eligibility_trace.ix[s, :] *= 0
        # self.eligibility_trace.ix[s, a] = 1

        # Q更新
        self.q_table += self.lr * error * self.eligibility_trace

        # 更新后的衰减资格迹
        if a_flag:
            self.eligibility_trace *= self.gamma*self.lambda_
        else:
            self.eligibility_trace *= 0
