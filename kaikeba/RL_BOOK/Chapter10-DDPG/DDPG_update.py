#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0
@license: Apache Licence
@software: PyCharm
@file: RL_brain.py
@time: 2018/6/23 8:17
@description: 更新版DDPG算法
"""

import numpy as np
import gym
from RL_brain import DDPG

#####################  全局参数  ####################

MAX_EPISODES = 200  # 最大回合数
MAX_EP_STEPS = 200  # 每个回合最大步数
MEMORY_CAPACITY = 10000 # 记忆库容量

RENDER = False  # 是否渲染环境
ENV_NAME = 'Pendulum-v0'    # 环境名称

###############################  训练  ####################################

env = gym.make(ENV_NAME)    # 加载环境
env = env.unwrapped # 取消限制
env.seed(1) # 设置种子

s_dim = env.observation_space.shape[0]  # 状态空间
a_dim = env.action_space.shape[0]   # 行为空间
a_bound = env.action_space.high # 行为值上限

ddpg = DDPG(a_dim, s_dim, a_bound)  # 创建DDPG决策类

var = 3  # 控制探索
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # 增加探索时的噪音
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # 为行动选择添加随机性进行探索
        s_, r, done, info = env.step(a)

        # 将当前的状态,行为,回报,下一个状态存储到记忆库中
        ddpg.store_transition(s, a, r / 10, s_)

        # 达到记忆库容量的最大值
        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # 衰减动作随机性
            ddpg.learn()    # 开始学习

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            # 达到回合最大值且回合回报值大于-300,渲染环境
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:RENDER = True
            break