#!/usr/bin/env python
# encoding: utf-8

import sys

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.maze import Maze
from RL_brain import SarsaLambdaTable, QLambdaTable
import numpy as np

# METHOD = "SarsaLambda"

METHOD = "QLambda"

def get_action(q_table, state):
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


def get_policy(q_table, rows=6, cols=6, pixels=40, orign=20):
    policy = []

    for i in range(rows):
        for j in range(cols):
            # 求出每个各自的状态
            item_center_x, item_center_y = (j * pixels + orign), (i * pixels + orign)
            item_state = [item_center_x - 15.0, item_center_y - 15.0, item_center_x + 15.0, item_center_y + 15.0]

            # 如果当前状态为各终止状态,则值为-1
            if item_state in [env.canvas.coords(env.hell1), env.canvas.coords(env.hell2),
                              env.canvas.coords(env.hell3), env.canvas.coords(env.hell4), env.canvas.coords(env.oval)]:
                policy.append(-1)
                continue

            if str(item_state) not in q_table.index:
                policy.append((0, 1, 2, 3))
                continue

            # 选择最优行为
            item_action_max = get_action(q_table, str(item_state))

            policy.append(item_action_max)

    return policy


# 判断当前状态是否在二级风道
def judge(observation):
    # 求出中心点坐标
    x = (observation[0] + observation[2]) / 2

    # 当横坐标为140时为风道
    if x == 140:
        return True
    return False


def update():
    for episode in range(1000):
        # 界面重置
        observation = env.reset()

        # 基于当前状态使用当前的策略选择的行为
        action = RL.choose_action(str(observation))

        # 初始化所有资格迹为0
        RL.eligibility_trace *= 0

        while True:
            # 刷新界面
            env.render()

            # 在风格子世界中,二级风的位置向上走会走两格,现在进行处理
            # 判断当前状态是否在二级风道且产生的动作为向上的动作
            if judge(observation) and action == 0:
                # 符合条件后,在本次循环中额外加一次向上运行的操作
                observation_, reward, done, oval_flag = env.step(action)

                # 如果过程中出现终止状态,直接结束
                if done:
                    break

                # 直接赋值为继续向上,回报不减少
                action_ = 0
                reward = 0.1

                # 从当前的改变中进行学习
                RL.learn(str(observation), action, reward, str(observation_), action_)

                # 修改当前的状态和行为
                observation = observation_
                action = action_

            # 从当前的状态采取行为得到下一个状态,回报,结束标志,宝藏位置标志
            observation_, reward, done, oval_flag = env.step(action)

            # 基于下一个状态选择行为
            action_ = RL.choose_action(str(observation_))

            # 如果在风道向下走会对没到陷阱时的情况做特殊处理(防止在风道进行来回增加回报)
            if judge(observation) and action == 1:
                reward = -0.1

            # 从当前的改变中进行学习
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 修改当前的状态和行为
            observation = observation_
            action = action_

            # 当达到终止条件时结束循环
            if done:
                break

    # 结束游戏
    print('游戏结束')

    # 开始输出最终的Q表
    q_table_result = RL.q_table

    # 使用Q表输出各状态的最优策略
    policy = get_policy(q_table_result)

    print("最优策略为", end=":")
    print(policy)

    print("迷宫格式为", end=":")
    policy_result = np.array(policy).reshape(6,6)
    print(policy_result)

    print("根据求出的最优策略画出方向")

    env.render_by_policy_new(policy_result)


if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    if METHOD == "QLambda":
        RL = QLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
