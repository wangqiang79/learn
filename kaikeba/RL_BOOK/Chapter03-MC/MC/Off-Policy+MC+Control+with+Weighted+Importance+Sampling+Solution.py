
# coding: utf-8
#geling 修改注释20180424
import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.halften import HalftenEnv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

env = HalftenEnv()

# 画出的图形中文乱码时的解决方案
matplotlib.rcParams['fonts.sans-serif'] = ['SimHei']
matplotlib.rcParams['fonts.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

xmajorLocator   = MultipleLocator(0.5)  # 刻度设置为0.5的倍数
xmajorFormatter = FormatStrFormatter('%1.1f') #设置x轴标签文本的格式

ymajorLocator   = MultipleLocator(1)
ymajorFormatter = FormatStrFormatter('%d') #设置y轴标签文本的格式

zmajorLocator   = MultipleLocator(1)
zmajorFormatter = FormatStrFormatter('%d') #设置z轴标签文本的格式

# 画图函数
figureIndex = 0
def prettyPrint(data, tile, zlabel='回报'):
    global figureIndex
    fig = plt.figure(figureIndex)
    figureIndex += 1
    fig.suptitle(tile)
    fig.set_size_inches(18.5, 10.5) # 调整输出的图像大小,因为刻度划分较细,所以使用默认图像大小时刻度会重叠显示,看不清效果

    ax = fig.add_subplot(111, projection='3d')
    axisX = []
    axisY = []
    axisZ = []

    ax.set_xlim(0.5, 10.5)  # 设置x轴刻度范围
    ax.set_ylim(1,5)        # 设置y轴刻度范围
    ax.set_zlim(0,1)        # 设置z轴刻度范围

    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)

    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)

    ax.zaxis.set_major_locator(zmajorLocator)
    ax.zaxis.set_major_formatter(zmajorFormatter)

    for i in data:
        axisX.append(i['x'])
        axisY.append(i['y'])
        axisZ.append(i['z'])
    ax.scatter(axisX, axisY, axisZ)
    ax.set_xlabel('玩家手牌总分')
    ax.set_ylabel('玩家手牌数')
    ax.set_zlabel(zlabel)

def create_random_policy(nA):
    """
    创建一个随机策略函数
    
    参数:
        nA: 这个环境中的行为数量
    
    返回:
        一个函数,参数为当前的状态,返回为一个可能的行为向量
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn

def create_greedy_policy(Q):
    """
    基于Q值生成一个贪心策略
    
    参数:
        Q: 一个字典，键为状态,值为动作
        
    Returns:
        一个函数,参数为当前的状态,返回为一个可能的行为向量
    """
    
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    使用重采样的蒙特卡罗离线策略控制
    找到最优的贪心策略

    参数:
        env: 十点半环境
        num_episodes: 对样本的自行次数
        behavior_policy: 行为策略
        discount_factor: 折扣因子
    
    Returns:
        A 元组 (Q, policy).
        Q 一个字典，键为状态,值为动作
        policy 一个函数,参数为当前的状态,返回为一个可能的行为向量
        这是一个最优的贪心策略
    """

    # 一个字典，键为状态, 值为动作
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # 加权重要性抽样公式的累积分母(通过所有的episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 遵循的策略
    target_policy = create_greedy_policy(Q)

    # 玩的场次
    for i_episode in range(1, num_episodes + 1):
        # 处理进度(每1000次在控制台更新一次)
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 定义一个episode数组,用来存入(state, action, reward)
        episode = []
        state = env._reset()
        for t in range(100):
            # 根据当前的状态返回一个可能的行为概率数组
            probs = behavior_policy(state)
            # 根据返回的行为概率随机选择动作
            action = np.random.choice(np.arange(len(probs)), p=probs)
            # 根据当前动作确认下一步的状态,回报,以及是否结束
            next_state, reward, done, _ = env._step(action)
            # 将当前的轨迹信息加入episode数组中
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        episode_len = len(episode)

        if episode_len < 4:
            continue
        
        # 累计回报的值
        G = 0.0
        # 权重
        W = 1.0

        print("局数:%d"%(i_episode))

        print("该局的episode",end=":")
        print(episode)

        # 对于每一个episode,倒序进行
        for t in range(len(episode))[::-1]:
            print("第%d次计算"%t)

            state, action, reward = episode[t]

            print(episode[t])

            # 从当前步更新总回报
            G = discount_factor * G + reward

            print("G更新",end=":")
            print(G)

            # 更新加权重要性采样公式分母
            C[state][action] += W

            print("C更新",end=":")

            print(C[state][action])

            # 使用增量更新公式更新动作值函数
            # 这也改善了我们提到Q的目标政策
            Q[state][action] += (W / C[state][action]) * G - Q[state][action]
            print("Q更新",end=":")

            print(Q[state][action])

            # 如果行为策略采取的行动不是目标策略采取的行动，那么概率将为0，我们可以打破
            if action != np.argmax(target_policy(state)):
                print("行为策略采取的行动与初始1策略不符,终止执行")
                print("*" * 50)
                break
            W = W * 1./behavior_policy(state)[action]
            print("W更新:",end=":")
            print(W)

            print("*"*50)

    return Q, target_policy

random_policy = create_random_policy(env.action_space.n)
Q, policy = mc_control_importance_sampling(env, num_episodes=500, behavior_policy=random_policy)

# policy_content = ["停牌","叫牌"]
#
# # 当手上有x张人牌时,各情况的最优策略
# action_0_pcard = []
# action_1_pcard = []
# action_2_pcard = []
# action_3_pcard = []
# action_4_pcard = []
#
# for state, actions in Q.items():
#     # 该状态下的最优行为值函数
#     action_value = np.max(actions)
#     # 该状态下的最优行为
#     best_action = np.argmax(actions)
#
#     score, card_num, p_num = state
#
#     item_0 = {"x": score, "y": int(card_num), "z": best_action}
#
#     if p_num == 0:
#         action_0_pcard.append(item_0)
#     elif p_num == 1:
#         action_1_pcard.append(item_0)
#     elif p_num == 2:
#         action_2_pcard.append(item_0)
#     elif p_num == 3:
#         action_3_pcard.append(item_0)
#     elif p_num == 4:
#         action_4_pcard.append(item_0)
#
#     print("当前手牌数之和为:%.1f,当前手牌数为:%d时,当前人牌数为%d,最优策略为:%s"%(score,card_num,p_num,policy_content[best_action]))
#
# prettyPrint(action_0_pcard, "没有人牌时的最优策略","采取策略")
# prettyPrint(action_1_pcard, "一张人牌时的最优策略","采取策略")
# prettyPrint(action_2_pcard, "两张人牌时的最优策略","采取策略")
# prettyPrint(action_3_pcard, "三张人牌时的最优策略","采取策略")
# prettyPrint(action_4_pcard, "四张人牌时的最优策略","采取策略")
# plt.show()


