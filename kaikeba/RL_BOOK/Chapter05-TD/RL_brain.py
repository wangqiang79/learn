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

    def learn(self, *args):
        pass

# 离线策略Q-learning
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            # 使用公式：Q_target = r+γ  maxQ(s',a')
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r

        # 更新公式: Q(s,a)←Q(s,a)+α(r+γ  maxQ(s',a')-Q(s,a))
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

# 在线策略Sarsa
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            # 使用公式: Q_taget = r+γQ(s',a')
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r
        # 更新公式: Q(s,a)←Q(s,a)+α(r+γQ(s',a')-Q(s,a))
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
