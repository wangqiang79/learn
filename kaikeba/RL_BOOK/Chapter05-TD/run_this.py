import sys
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.maze import Maze
from RL_brain import QLearningTable,SarsaTable

METHOD = "Q-Learning"

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
                episode_num = episode
                step_num += c

                print(policy)
                print("*"*50)

                # 如果N次行走的策略相同,表示已经收敛
                if policy == tmp_policy and oval_flag:
                    count = count + 1

                    if count == N:
                        flag = True
                else:
                    count = 0

                    policy = tmp_policy
                break

        if flag:
            break

    if flag:
        print("="*50)

        print('算法%s在%s局时收敛,总步数为:%d'%(METHOD,episode_num,step_num))
        print('最优策略输出',end=":")
        print(policy)

        # 在界面上进行展示
        env.reset()
        env.render_by_policy(policy)

    else:
        # 达到设置的局数,终止游戏
        print('算法%s未收敛,但达到了100局,游戏结束'%METHOD)
        env.destroy()

if __name__ == "__main__":
    env = Maze()

    RL = SarsaTable(actions=list(range(env.n_actions)))

    if METHOD =="Q-Learning":
        RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()