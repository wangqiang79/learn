#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/26 16:03
# @Author  : liuyb
# @Site    : 
# @File    : Off_On_Poilcy_Compare.py
# @Software: PyCharm
# @Description: 蒙特卡罗在线离线策略值函数对比

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.halften import HalftenEnv

env = HalftenEnv()

##################################在线策略部分##################################
# 返回一个epsilon贪心策略函数
def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        # 在状态空间中求最大行为值函数
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn

# 在线策略
def on_policy(state,Q,discount_factor,returns_sum,returns_count,episode,policy):
    for t in range(100):
        # 根据当前的状态返回一个可能的行为概率数组
        probs = policy(state)
        # 根据返回的行为概率随机选择动作
        action = np.random.choice(np.arange(len(probs)), p=probs)
        # 根据当前动作确认下一步的状态,回报,以及是否结束
        next_state, reward, done, _ = env._step(action)
        # 将当前的轨迹信息加入episode数组中
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    # 从所有轨迹中提取出(state,action)
    sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
    for state, action in sa_in_episode:
        sa_pair = (state, action)
        # 使用初访法统计累计回报的均值
        # 找到状态,动作在所有轨迹中第一次出现的索引
        first_occurence_idx = next(i for i, x in enumerate(episode)
                                   if x[0] == state and x[1] == action)
        # 从第一次出现的位置起计算累计回报
        G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
        # 计算当前状态的累计回报均值
        returns_sum[sa_pair] += G
        returns_count[sa_pair] += 1.0
        # 策略的提升就是不断改变该状态下的Q
        Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

    return Q,policy,returns_sum,returns_count,episode

##################################离线策略部分##################################
def create_random_policy(nA):
    A = np.ones(nA, dtype=float) / nA

    def policy_fn(observation):
        return A

    return policy_fn

def create_greedy_policy(Q):
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn

# 离线策略
def off_policy(state,behavior_policy,episode,discount_factor,target_policy,C,Q):
    for t in range(100):
        # 根据当前的状态返回一个行为概率数组
        probs = behavior_policy(state)
        # 根据返回的行为概率数组随机选择动作
        action = np.random.choice(np.arange(len(probs)), p=probs)
        # 根据当前动作确认下一步的状态,回报,以及是否结束
        next_state, reward, done, _ = env._step(action)
        # 将当前的轨迹信息加入episode数组中
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    # 累计回报的值
    G = 0.0
    # 权重
    W = 1.0
    # 对于每一个episode,倒序进行计算
    for t in range(len(episode))[::-1]:
        state, action, reward = episode[t]
        # 从当前步更新总回报
        G = discount_factor * G + reward
        # 更新加权重要性采样公式分母
        C[state][action] += W
        # 使用增量更新公式更新动作值函数
        Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
        # 如果行为策略采取的行动不是目标策略采取的行动，则跳出循环
        if action != np.argmax(target_policy(state)):
            break
        W = W * 1. / behavior_policy(state)[action]

    return Q,target_policy,C,episode

def play(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    #####################在线策略部分的迭代变量#####################
    # 在线策略迭代变量
    # 返回的轨迹累计回报之和
    returns_sum = defaultdict(float)
    # 返回的所有轨迹的数量
    returns_count = defaultdict(float)

    # 最终的行为空间
    Q_on = defaultdict(lambda: np.zeros(env.action_space.n))

    # 遵循的策略
    policy = make_epsilon_greedy_policy(Q_on, epsilon, env.action_space.n)

    #####################离线策略部分的迭代变量#####################
    # 行为值函数
    Q_off = defaultdict(lambda: np.zeros(env.action_space.n))
    # 加权重要采样公式的累积分母(通过所有的episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    random_policy = create_random_policy(env.action_space.n)

    # 需要评估改善的目标策略为贪心策略
    target_policy = create_greedy_policy(Q_off)

    # 玩的局数
    for i_episode in range(1, num_episodes + 1):
        # 处理进度(每1000次在控制台更新一次)
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 定义一个episode数组,用来存入(state, action, reward)
        episode_on = []
        episode_off = []
        # 游戏开始
        state = env._reset()

        # 在线策略
        Q_on, policy, returns_sum, returns_count, episode_on = on_policy(state, Q_on, discount_factor, returns_sum, returns_count, episode_on, policy)

        # 离线策略
        Q_off, target_policy, C, episode_off = off_policy(state, random_policy, episode_off, discount_factor, target_policy, C, Q_off)

    return Q_on, policy,Q_off,target_policy

Q_on, policy,Q_off,target_policy = play(env, num_episodes=500000, epsilon=0.1)

print("")

result_on = []
result_off = []

for state, actions in Q_on.items():
    # 该状态下的最优行为值函数
    action_value = np.max(actions)
    # 该状态下的最优行为
    best_action = np.argmax(actions)

    score, card_num, p_num = state

    item = {"x": score, "y": int(card_num), "z": best_action, "p_num":p_num}

    result_on.append(item)

for state, actions in Q_off.items():
    # 该状态下的最优行为值函数
    action_value = np.max(actions)
    # 该状态下的最优行为
    best_action = np.argmax(actions)

    score, card_num, p_num = state

    item = {"x": score, "y": int(card_num), "z": best_action, "p_num":p_num}

    result_off.append(item)

# 排序
result_on.sort(key=lambda obj:obj.get('x'), reverse=False)
result_off.sort(key=lambda obj:obj.get('x'), reverse=False)

for on, off in zip(result_on, result_off):
    print(on,end="==>")
    print(off)
