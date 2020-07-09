import sys
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.maze import Maze
from RL_brain import QLearningTable,SarsaTable
import numpy as np

METHOD = "Q-Learning"

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

def update():
    for episode in range(100):
        # 初始化状态
        observation = env.reset()

        c = 0

        tmp_policy = {}

        while True:
            # 渲染当前环境
            env.render()

            # 基于当前状态选择行为
            action = RL.choose_action(str(observation))

            state_item = tuple(observation)

            tmp_policy[state_item] = action

            # 采取行为获得下一个状态和回报,及是否终止
            observation_, reward, done, oval_flag = env.step(action)

            if METHOD == "SARSA":
                # 基于下一个状态选择行为
                action_ = RL.choose_action(str(observation_))

                # 基于变化 (s, a, r, s, a)使用Sarsa进行Q的更新
                RL.learn(str(observation), action, reward, str(observation_), action_)
            elif METHOD == "Q-Learning":
                # 根据当前的变化开始更新Q
                RL.learn(str(observation), action, reward, str(observation_))

            # 改变状态和行为
            observation = observation_

            c += 1

            # 如果为终止状态,结束当前的局数
            if done:
                break

    print('游戏结束')

    # 开始输出最终的Q表
    q_table_result = RL.q_table

    # 使用Q表输出各状态的最优策略
    policy = get_policy(q_table_result)

    print("最优策略为",end=":")
    print(policy)

    print("迷宫格式为",end=":")
    policy_result = np.array(policy).reshape(5,5)
    print(policy_result)

    print("根据求出的最优策略画出方向")

    env.render_by_policy_new(policy_result)

    # env.destroy()

if __name__ == "__main__":
    env = Maze()

    RL = SarsaTable(actions=list(range(env.n_actions)))

    if METHOD =="Q-Learning":
        RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()