
# coding: utf-8
#geling 修改注释20180424
# In[1]:

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
    fig.set_size_inches(18.5, 10.5)  # 调整输出的图像大小,因为刻度划分较细,所以使用默认图像大小时刻度会重叠显示,看不清效果

    ax = fig.add_subplot(111, projection='3d')
    axisX = []
    axisY = []
    axisZ = []

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(1,5)
    ax.set_zlim(0,1)

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

# 返回一个epsilon贪心策略函数
def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    基于给定的Q函数和epsilon创建一个epsilon贪心策略

    参数:
        行为值函数Q：一个字典,键为状态空间(状态 -> 行为值函数数组),其中行为值函数为一个长度为2的数组
        参数epsilon: 选取一个随机动作的可能性,值介于0和1之间
        行为数nA: 行为数
    
    返回:
        求取最优策略的函数
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        # 在状态空间中求最大行为值函数
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

# 蒙特卡罗控制使用epsilon贪心策略
def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    使用epsilon贪心策略的模特卡罗控制,找到最优的epsilon贪心策略
    
    参数:
        env: 十点半环境.
        num_episodes: 样本的总步数.
        discount_factor: 衰减因子.
        epsilon: 随机选择动作的概率值,在0和1之间
    
    返回:
        A tuple (Q, policy).
        Q 是一个字典,键为状态,值为值函数集合
        policy 是一个根据传入状态返回最可能进行下一步行为的策略函数
    """
    # 返回的轨迹累计回报之和
    returns_sum = defaultdict(float)
    # 返回的所有轨迹的数量
    returns_count = defaultdict(float)
    
    # 最终的行为空间
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 遵循的策略
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # 玩的局数
    for i_episode in range(1, num_episodes + 1):
        # 处理进度(每1000次在控制台更新一次)
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 定义一个episode数组,用来存入(state, action, reward)
        episode = []
        # 游戏开始
        state = env._reset()

        for t in range(100):
            # 根据当前的状态返回一个可能的行为概率数组
            probs = policy(state)
            # 根据返回的行为概率随机选择动作
            action = np.random.choice(np.arange(len(probs)), p=probs)
            # 根据当前动作确认下一步的状态,回报,以及是否结束
            next_state, reward, done, _ = env._step(action)
            # 将当前的轨迹信息加入episode数组中
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # 从所有轨迹中提取出(state,action)
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # 使用初访法统计累计回报的均值
            # 找到状态,动作在所有轨迹中第一次出现的索引
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # 从第一次出现的位置起计算累计回报
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # 计算当前状态的累计回报均值
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            # 策略的提升就是不断改变该状态下的Q
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
    return Q, policy

Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

print("")

policy_content = ["停牌","叫牌"]

# 当手上有x张人牌时,各情况的最优策略
action_0_pcard = []
action_1_pcard = []
action_2_pcard = []
action_3_pcard = []
action_4_pcard = []

result = []

for state, actions in Q.items():
    # 该状态下的最优行为值函数
    action_value = np.max(actions)
    # 该状态下的最优行为
    best_action = np.argmax(actions)

    score, card_num, p_num = state

    item = {"x": score, "y": int(card_num), "z": best_action, "p_num":p_num}

    result.append(item)

    # if p_num == 0:
    #     action_0_pcard.append(item)
    # elif p_num == 1:
    #     action_1_pcard.append(item)
    # elif p_num == 2:
    #     action_2_pcard.append(item)
    # elif p_num == 3:
    #     action_3_pcard.append(item)
    # elif p_num == 4:
    #     action_4_pcard.append(item)
    #
    # print("当前手牌数之和为:%.1f,当前手牌数为:%d时,当前人牌数为%d,最优策略为:%s"%(score,card_num,p_num,policy_content[best_action]))

# prettyPrint(action_0_pcard, "没有人牌时的最优策略","采取策略")
# prettyPrint(action_1_pcard, "一张人牌时的最优策略","采取策略")
# prettyPrint(action_2_pcard, "两张人牌时的最优策略","采取策略")
# prettyPrint(action_3_pcard, "三张人牌时的最优策略","采取策略")
# prettyPrint(action_4_pcard, "四张人牌时的最优策略","采取策略")
# plt.show()

result.sort(key=lambda obj:obj.get('x'), reverse=False)

for tmp in result:
    score = tmp["x"]
    card_num = tmp["y"]
    z = tmp["z"]
    p_num = tmp["p_num"]
    print("当前手牌数之和为:%.1f,当前手牌数为:%d时,当前人牌数为%d,最优策略为:%s" % (score, card_num, p_num, policy_content[z]))

