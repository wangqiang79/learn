# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow
Tested in Tensorflow 1.4 and 1.5

@author: Xiang Zhong
"""

import numpy as np
import tensorflow as tf


class PolicyValueNet():
    '''
    策略价值网络
    '''
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height

        # 1. 输入: 当前的4个局面,4x15x15
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, 4, board_height, board_width])
        # 转换输入变成15x15x4
        self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])
        # 2. 通用的网络层
        # conv1: 3x3x32 SAME RELE   -- 15x15x32
        self.conv1 = tf.layers.conv2d(inputs=self.input_state,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", data_format="channels_last",
                                      activation=tf.nn.relu)
        # conv2: 3x3x32 SAME RELU   -- 15x15x64
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        # conv3: 3x3x128 SAME RELU  -- 15x15x128
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        # 动作网络
        # conv: 1x1x4 SAME RELU -- 15x15x4
        self.action_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            data_format="channels_last",
                                            activation=tf.nn.relu)
        # 拉平
        # 900
        self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, 4 * board_height * board_width])
        # 全连接层, 输出的是棋盘上每个位置移动的对数概率
        # 神经元: 15x15 log_softmax
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)
        # 评估网络
        # evaluation_conv: 1x1x2 SAME RELU
        # 输入为conv3的输出: 15x15x4  — 15x15x2
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                data_format="channels_last",
                                                activation=tf.nn.relu)
        # 拉平: 450
        self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, 2 * board_height * board_width])
        # 全连接层: 神经元: 64 RELU	— 64
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # 输出当前状态的评估分数
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)

        # 定义损失函数
        # 1. 标签数组: 包含每个状态是否获胜
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. 预测数组: 包含每个状态的评估分数
        # 3-1. 值损失函数
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc2)
        # 3-2. 策略损失函数
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        # 3-3. L2正则化部分
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 损失函数: 值损失函数 + 策略损失函数 + L2正则化
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # 定义优化器
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # 创建Session
        self.session = tf.Session()

        # 计算策略熵，仅用于监控
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # 初始化变量
        init = tf.global_variables_initializer()
        self.session.run(init)

        # 如果模型存在,加载模型文件
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        """
        计算策略值
        输入: 一个批次的状态数组
        输出: 行为概率和状态值
        """
        log_act_probs, value = self.session.run(
                [self.action_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        """
        策略值函数
        输入: 棋盘
        输出: 每个可用操作的(动作，概率)元组列表以及棋盘状态的分数
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """
        执行训练步骤
        """
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        '''
        存储模型文件
        :param model_path:
        :return:
        '''
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        '''
        加载模型文件
        :param model_path:
        :return:
        '''
        self.saver.restore(self.session, model_path)
