#!/usr/bin/env python  
# encoding: utf-8  

""" 
@version: v1.0 
@license: Apache Licence
@software: PyCharm 
@file: RL_brain.py
@time: 2018/6/23 8:17
@description: DDPG决策类
"""
import tensorflow as tf
import numpy as np

LR_A = 0.001  # 演员网络学习率
LR_C = 0.001  # 评论家网络学习率
GAMMA = 0.9  # 回报的折扣因子
TAU = 0.01      # 简单替换
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # 设置记忆库存储的结构
        self.pointer = 0    # 记忆库当前容量
        self.sess = tf.Session()    # tf session会话
        # self.a_replace_counter, self.c_replace_counter = 0, 0   # 演员网络替换次数,评论家网络替换次数

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,   # 动作空间,状态空间,动作范围
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's') # 当前状态预留占位符
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')   # 下一状态预留占位符
        self.R = tf.placeholder(tf.float32, [None, 1], 'r') # 回报预留占位符

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)    # 演员-评估网络根据当前状态输出行为
            a_ = self._build_a(self.S_, scope='target', trainable=False)    # 演员-目标网络根据下一个状态输出行为
        with tf.variable_scope('Critic'):
            # 当为td_error计算q时，在内存中分配self.a = a，否则当更新Actor时self.a来自Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True) # 根据当前状态和来自演员-评估网络的行为,计算q值
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)    # 根据下一状态和来自演员-目标网络的行为,计算q_值

        # 网络参数
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # 目标网络参数替换(简单替换)
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # 在td_error的feed_dic中，self.a应该更改为内存中的行为
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)  # 根据td_error进行更新评论家网络

        # 根据为代码进行修改: 相当于dq / da * da / dparams,即可以直接对q求梯度即可
        self.policy_grads = tf.gradients(ys=self.a, xs=self.ae_params, grad_ys=tf.gradients(q, self.a)[0])
        self.atrain = tf.train.AdamOptimizer(-LR_A).apply_gradients(zip(self.policy_grads, self.ae_params))

        # a_loss = - tf.reduce_mean(q)
        # self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)    # 取q的最大值进行更新演员网络

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # 简单目标网络参数的替换
        self.sess.run(self.soft_replace)

        # 随机选择记忆库中BATCH_SIZE各数据进行更新
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # 演员网络更新
        self.sess.run(self.atrain, {self.S: bs})
        # 评论家网络更新
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # 使用新的记忆替换旧的
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)