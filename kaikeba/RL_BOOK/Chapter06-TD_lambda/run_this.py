#!/usr/bin/env python
# encoding: utf-8

import sys
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.maze import Maze
from RL_brain import SarsaLambdaTable,QLambdaTable

METHOD = "SarsaLambda"
# METHOD = "QLambda"

# 判断当前状态是否在二级风道
def judge(observation):
    # 求出中心点坐标
    x = (observation[0] + observation[2]) / 2

    # 当横坐标为140时为风道
    if x == 140:
        return True
    return False

def update():
    # 收敛标记
    flag = False
    # 连续N次达到宝藏位置,即为收敛
    N = 3

    # 相似次数
    count = 0

    # 初始化一个随机策略
    policy = {}

    # 记录局数
    episode_num = 0
    # 记录总步数
    step_num = 0

    result_list = []

    for episode in range(100):
        # 界面重置
        observation = env.reset()

        # 基于当前状态使用当前的策略选择的行为
        action = RL.choose_action(str(observation))

        # 初始化所有资格迹为0
        RL.eligibility_trace *= 0

        c = 0

        tmp_policy = {}

        # 因为连线需要顺序,所以,现在使用列表表示每步的策略
        tmp_list = []

        while True:
            # 刷新界面
            env.render()

            tmp_policy[tuple(observation)] = action

            tmp_list.append(tuple(observation))

            # 在风格子世界中,二级风的位置向上走会走两格,现在进行处理
            # 判断当前状态是否在二级风道且产生的动作为向上的动作
            if judge(observation) and action == 0:
                # 符合条件后,在本次循环中额外加一次向上运行的操作
                observation_, reward, done, oval_flag = env.step(action)
                # 直接赋值为继续向上,回报不减少
                action_ = 0
                reward = 0

                # 从当前的改变中进行学习
                RL.learn(str(observation), action, reward, str(observation_), action_)

                # 修改当前的状态和行为
                observation = observation_
                action = action_

                # 如果过程中出现终止状态,直接结束
                if done:
                    episode_num = episode
                    step_num += c

                    print(policy)
                    print("*" * 50)

                    count = 0

                    policy = tmp_policy

                    break

            # 从当前的状态采取行为得到下一个状态,回报,结束标志,宝藏位置标志
            observation_, reward, done, oval_flag = env.step(action)

            # 基于下一个状态选择行为
            action_ = RL.choose_action(str(observation_))

            # 从当前的改变中进行学习
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 修改当前的状态和行为
            observation = observation_
            action = action_

            c += 1

            # 当达到终止条件时结束循环
            if done:
                episode_num = episode
                step_num += c

                print(policy)
                print("*" * 50)

                # 如果N次行走的策略相同,表示已经收敛
                if policy == tmp_policy and oval_flag:
                    count = count + 1

                    if count == N:
                        result_list = tmp_list

                        flag = True
                else:
                    count = 0

                    policy = tmp_policy
                break
        if flag:
            break

    # 结束游戏
    # print('game over')
    # env.destroy()
    if flag:
        print("="*50)

        print('算法%s在%s局时收敛,总步数为:%d'%(METHOD,episode_num,step_num))
        print('最优策略输出',end=":")
        print(policy)

        # 在界面上进行展示
        env.reset()
        env.render_by_policy(policy,result_list)
    else:
        # 达到设置的局数,终止游戏
        print('算法%s未收敛,但达到了100局,游戏结束'%METHOD)
        env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    if METHOD == "QLambda":
        RL = QLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()