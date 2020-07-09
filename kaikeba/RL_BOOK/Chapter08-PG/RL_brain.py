'''
    Policy Gradient 策略梯度的决策类
'''
import numpy as np
import tensorflow as tf

# 重复性
np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # 全连接层1
        layer = tf.layers.dense(
            inputs=self.tf_obs,  # 该层的输入
            units=10,  # 神经元数量
            activation=tf.nn.tanh,  # 激活函数
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            # 用于权重矩阵的初始化函数。 如果无（默认），则使用tf.get_variable使用的默认初始化程序初始化权重
            bias_initializer=tf.constant_initializer(0.1),  # 初始化偏置量
            name='fc1'  # 该层的名称
        )
        # 全连接层2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # 为使总回报最大化（log_p * R）即为最小化的 -(log_p * R），并且tf仅具有最小化（损失）
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
                                                                          labels=self.tf_acts)  # this is negative log of chosen action
            # 上边的十字还可以用下面的式子表示
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # 更新公式θ<--θ+α▽_θlogπ_θ(s_t,a_t)v_t

        with tf.name_scope('train'):
            # 根据损失进行优化操作
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob,
                                     feed_dict={self.tf_obs: observation[np.newaxis, :]})  # np.newaxis 在使用和功能上等价于 None
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # 折扣和规范化每一个ep中的奖励
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # 对ep中的奖励进行折扣
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # 反序遍历
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 规范化回报值
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
