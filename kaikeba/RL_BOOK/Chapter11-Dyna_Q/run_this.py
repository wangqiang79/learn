#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/26 15:24
# @Site    :
# @File    : __init__.py.py
# @Software: PyCharm
# @Description: 使用Dyna-Q框架进行迷宫寻宝

from lib.envs.maze import Maze
from RL_brain import QLearningTable, EnvModel
import matplotlib.pyplot as plt

import numpy as np

def get_action(q_table,state):
    # 选择最优行为
    state_action = q_table.ix[state, :]

    # 因为一个状态下最优行为可能会有多个,所以在碰到这种情况时,需要随机选择一个行为进行
    state_action_max = state_action.max()

    idxs = []

    for max_item in range(len(state_action)):
        if state_action[max_item] == state_action_max:
            idxs.append(max_item)

    sorted(idxs)
    return tuple(idxs)

def get_policy(q_table,rows=5,cols=5,pixels=40,orign=20):
    policy = []

    for i in range(rows):
        for j in range(cols):
            # 求出每个各自的状态
            item_center_x, item_center_y = (j * pixels + orign), (i * pixels + orign)
            item_state = [item_center_x - 15.0, item_center_y - 15.0, item_center_x + 15.0, item_center_y + 15.0]

            # 如果当前状态为各终止状态,则值为-1
            if item_state in [env.canvas.coords(env.hell1), env.canvas.coords(env.hell2),
                                   env.canvas.coords(env.hell3), env.canvas.coords(env.hell4),
                                   env.canvas.coords(env.hell5), env.canvas.coords(env.hell6),
                                   env.canvas.coords(env.hell7), env.canvas.coords(env.oval)]:
                policy.append(-1)
                continue

            if str(item_state) not in q_table.index:
                policy.append((0, 1, 2, 3))
                continue

            # 选择最优行为
            item_action_max = get_action(q_table,str(item_state))

            policy.append(item_action_max)

    return policy

# 绘制规划次数和收敛步数的关系
def show_plot(result):
    plt.figure(figsize=(15, 4))

    for planning_step in result:
        plt.plot(result[planning_step].keys(),result[planning_step].values(), label='%s planning steps'%planning_step)

    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.title('DynaQ Algorithm Maze')

    plt.show()

def update():
    # 当前设置规划次数为0,5和50次
    planning_steps = [0,5,50]

    for planning_step in planning_steps:
        result[planning_step] = {}
        # 将q表清空
        RL.clear_q_table()
        # 将仿真环境中的记忆库清空
        env_model.clear_database()

        for episode in range(50):
            # 初始化环境得到当前状态
            s = env.reset()

            # 统计当前回合的步数
            step = 0

            while True:
                env.render()
                # 根据当前状态选择行为
                a = RL.choose_action(str(s))
                # 从环境中获取系一步的状态,回报和终止标识
                s_, r, done,oval_flag = env.step(a)
                # Q-Learning模型开始进行学习更新
                RL.learn(str(s), a, r, str(s_))

                # 通过输入（s，a）使用模型输出（r，s_）
                # dyna Q版本的模型就像内存重播缓冲区一样
                env_model.store_transition(str(s), a, r, s_)
                for n in range(planning_step):     # 使用env_model再学习planning_step次
                    # 从输入的数据库中随机选择状态和动作
                    ms, ma = env_model.sample_s_a()
                    # 通过状态和动作获得回报和下一个状态
                    mr, ms_ = env_model.get_r_s_(ms, ma)
                    # 使用Q-Learning算法再进行学习更新
                    RL.learn(ms, ma, mr, str(ms_))

                s = s_

                step += 1
                if done:
                    # 存储当前回合和步数
                    result[planning_step][episode] = step
                    break

    print('游戏结束')

    # 开始输出最终的Q表
    q_table_result = RL.q_table

    # 使用Q表输出各状态的最优策略
    policy = get_policy(q_table_result)

    print("最优策略为", end=":")
    print(policy)

    print("迷宫格式为", end=":")
    policy_result = np.array(policy).reshape(5, 5)
    print(policy_result)

    print("根据求出的最优策略画出方向")

    env.render_by_policy_new(policy_result)

    print(' 绘制规划次数和每回合收敛步数的关系图')
    # 2018-10-25新增需求: 绘制规划次数n和每回合收敛步数的关系图
    show_plot(result)

if __name__ == "__main__":
    # 2018-10-25需求添加: 绘制规划次数n和每回合收敛步数的关系图
    result = {}

    # 创建迷宫环境
    env = Maze()

    print(env.n_actions)
    print(list(range(env.n_actions)))
    # 创建Q-Learning决策对象
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # 创建环境模型
    env_model = EnvModel(actions=list(range(env.n_actions)))

    env.after(0, update)
    env.mainloop()