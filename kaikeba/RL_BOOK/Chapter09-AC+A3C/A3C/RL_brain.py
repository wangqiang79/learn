#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 14:40
# @Author  : liuyb
# @Site    : 
# @File    : RL_brain.py
# @Software: PyCharm
# @Description: A3C决策类

import tensorflow as tf
import numpy as np
import gym

GAME = 'Pendulum-v0'  # 环境名称
GLOBAL_NET_SCOPE = 'Global_Net'  # 全局网络的范围名称

MAX_EP_STEP = 200  # 每回合最大步数
MAX_GLOBAL_EP = 2000  # 最大回合数

UPDATE_GLOBAL_ITER = 10  # 全局网络更新频率
GAMMA = 0.9  # 回报的衰减值
ENTROPY_BETA = 0.01  # 熵值

GLOBAL_RUNNING_R = []  # 全局运行时的回报
GLOBAL_EP = 0  # 全局的回合次数

# 定义A-C网络(全局的A-C网络和本地的A-C网络处理稍有不同)
class ACNet(object):
    def __init__(self, scope, sess, n_s, n_a, a_bound, OPT_A = None, OPT_C = None, globalAC=None):
        self.sess = sess
        self.n_s = n_s
        self.n_a = n_a
        self.a_bound = a_bound

        if scope == GLOBAL_NET_SCOPE:
            # 全局网络(使用worker运行的网络)
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.n_s], 'S')  # 状态的占位符
                self.a_params, self.c_params = self._build_net(scope)[-2:]  # 创建A-C网络并得到Actor网络的参数和Critic网络的参数
        else:
            # 本地网络,根据各全局网络的反馈计算损失
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.n_s], 'S')  # 状态占位符
                self.a_his = tf.placeholder(tf.float32, [None, self.n_a], 'A')  # 动作占位符
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # 目标状态值的占位符

                '''
                scope为None时创建本地网络,返回
                mu:使用tanh激活函数算出的行为
                sigma:使用softplus激活函数算出的行为
                v:评论家网路输出的状态值函数
                a_params:演员网络的参数
                c_params:评论家网络的参数
              '''
                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                # 使用目标状态值和网络输出的状态值求出TD_error
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    # 计算评论家网络的损失
                    self.c_loss = tf.reduce_mean(tf.square(td))

                # 针对演员网络计算出的行为进行格式化
                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * self.a_bound[1], sigma + 1e-4

                # 将计算出的行为值进行格式化后放入分发模块中
                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    # 使用函数计算log_prob    使用更新公式θ<--θ+α▽_θlogπ_θ(s_t,a_t)v_t
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # 鼓励探索行为(这个函数的意思是将确定性行为变得随机,以便得到更好的结果)
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    # 计算演员网络的损失
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):
                    # 使用本地的参数选择行为
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), self.a_bound[0], self.a_bound[1])
                with tf.name_scope('local_grad'):
                    # 根据演员网络的损失计算梯度
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    # 根据评论家网络的损失计算梯度
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                # 从本地参数更新当前worker的A-C网络
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                # 向本地的A-C网络反馈当前worker中的A-C网络的情况
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, self.n_a, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, self.n_a, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # 状态值
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):
        # 本地网络运行
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # 将梯度应用到本地网络上

    def pull_global(self):
        # 本地网络运行
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])    #从本地网络获取更新后的A-C网络参数

    def choose_action(self, s):
        # 本地网络运行
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.s: s})[0]

class Worker(object):
    def __init__(self, name, globalAC, coord, sess, OPT_A = None, OPT_C = None):
        self.env = gym.make(GAME).unwrapped # 创建环境
        self.name = name    # worker名称
        self.AC = ACNet(name, sess, self.env.observation_space.shape[0], self.env.action_space.shape[0], [self.env.action_space.low, self.env.action_space.high], OPT_A,OPT_C, globalAC) # 创建名为name的AC网络
        self.coord = coord
        self.sess = sess

    def work(self):
        # 声明使用全局变量
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []   # 定义三个缓存数组 状态   行为  奖励
        while not self.coord.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                if self.name == 'W_0':
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == MAX_EP_STEP - 1 else False   # 当达到回合最大步数时强制终止

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # 因为返回的回报都一样,现在进行标准化

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # 更新全局并分配给本地网络
                    if done:
                        v_s_ = 0  # 终止状态
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # 反向遍历缓存r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    # 更新全局并分配给本地网络后,将缓存清空,并重新从本地网络获取A-C网络参数
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # 记录运行中的回合奖励
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        # 对奖励做一些特殊处理
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    GLOBAL_EP += 1
                    break

def get_global_reward():
    return GLOBAL_RUNNING_R