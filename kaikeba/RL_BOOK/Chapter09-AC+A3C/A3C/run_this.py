#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 14:51
# @Author  : liuyb
# @Site    : 
# @File    : run_this.py
# @Software: PyCharm
# @Description: A3C运行 Pendulum-v0游戏

import os
import numpy as np
import gym
import tensorflow as tf
import multiprocessing
import threading
import shutil
import matplotlib.pyplot as plt
import RL_brain
from RL_brain import ACNet,Worker

GAME = 'Pendulum-v0'  # 环境名称
OUTPUT_GRAPH = True  # 是否输出graph
LOG_DIR = './log'  # log文件夹路径

LR_A = 0.0001  # 演员网络的学习率
LR_C = 0.001  # 评论家网络的学习率
GLOBAL_NET_SCOPE = 'Global_Net'  # 全局网络的范围名称
N_WORKERS = multiprocessing.cpu_count()  # 根据CPU核数指定worker

env = gym.make(GAME)  # 创建环境

N_S = env.observation_space.shape[0]  # 状态空间
N_A = env.action_space.shape[0]  # 行为空间
A_BOUND = [env.action_space.low, env.action_space.high]  # 行为值的上下限

with tf.Session() as sess:
    # 创建一个协调器,管理线程
    COORD = tf.train.Coordinator()

    with tf.device("/cpu:0"):
        # 分别运行A和C网络效果好
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')

        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE,sess,N_S,N_A,A_BOUND,OPT_A,OPT_C)  # 我们只需要它的参数
        workers = []
        # 创建worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker命名
            # name, globalAC, coord, sess, OPT_A = None, OPT_C = None
            workers.append(Worker(i_name, GLOBAL_AC, COORD, sess, OPT_A, OPT_C))

    # 初始化所有参数
    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, sess.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    # 所有线程运行完毕之后再进行下面的操作
    COORD.join(worker_threads)

    global_reward = RL_brain.get_global_reward()

    plt.plot(np.arange(len(global_reward)), global_reward)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()