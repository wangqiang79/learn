# coding: utf-8
#geling修改注释 20180421
#liuyubiao修改策略输出为多策略输出
import numpy as np
import pprint
import sys
import PolicyEvaluationSolution

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

# 进行多策略的输出

# 定义两个全局变量用来记录运算的次数
v_num = 1
i_num = 1

# 根据传入的四个行为选择值函数最大的索引,返回的是一个索引数组和一个行为策略
def get_max_index(action_values):
    indexs = []
    policy_arr = np.zeros(len(action_values))

    action_max_value = np.max(action_values)

    for i in range(len(action_values)):
        action_value = action_values[i]

        if action_value == action_max_value:
            indexs.append(i)
            policy_arr[i] = 1
    return indexs,policy_arr

# 将策略中的每行可能行为改成元组形式,方便对多个方向的表示
def change_policy(policys):
    action_tuple = []

    for policy in policys:
        indexs, policy_arr = get_max_index(policy)
        action_tuple.append(tuple(indexs))

    return action_tuple
#policy_improvement是策略迭代方法，输入为环境env，策略评估函数policy_eval_，折扣因子。输出为最优值函数和最优策略
# policy_eval_fn=PolicyEvaluationSolution.policy_eval表示对PolicyEvaluationSolution文件的policy_eval方法进行调用。
def policy_improvement(env, policy_eval_fn=PolicyEvaluationSolution.policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # 初始化一个随机策略
    policy = np.ones([env.nS, env.nA]) / env.nA

    print("初始的随机策略")
    print(policy)
    print("*"*50)

    while True:
        global i_num
        global v_num

        v_num = 1
        # 评估当前的策略,输出为各状态的当前的状态值函数
        V = policy_eval_fn(policy, env, discount_factor)

        print("第%d次策略提升时求出的各状态值函数"%i_num)
        print(V)

        print("")
        
        # 定义一个当前策略是否改变的标识
        policy_stable = True
        
        # 遍历各状态
        for s in range(env.nS):
            # 取出当前状态下最优行为的索引值
            chosen_a = np.argmax(policy[s])
            
            # 初始化行为数组[0,0,0,0]
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                # 遍历各行为
                for prob, next_state, reward, done in env.P[s][a]:
                    # 根据各状态值函数求出行为值函数
                    action_values[a] += prob * (reward + discount_factor * V[next_state])

            # v1.0版更新内容,因为np.argmax(action_values)只会选取第一个最大值出现的索引,所以会丢掉其他方向的可能性,现在会输出一个状态下所有的可能性
            best_a_arr, policy_arr = get_max_index(action_values)
            
            # 如果求出的最大行为值函数的索引(方向)没有改变,则定义当前策略未改变，收敛输出
            # 否则将当前的状态中将有最大行为值函数的方向置1,其余方向置0
            if chosen_a not in best_a_arr:
                policy_stable = False
            policy[s] = policy_arr

        print("第%d次策略提升结果"%i_num)
        print(policy)
        print("*"*50)

        i_num = i_num + 1
        
        # 如果当前策略没有发生改变,即已经到了最优策略,返回
        if policy_stable:
            print("第%d次之后得到的结果已经收敛,运算结束"%(i_num-1))

            return policy, V


env = GridworldEnv()
policy, v = policy_improvement(env)
print("策略可能的方向值:")
print(policy)
print("")

print("策略网格形式 (0=up, 1=right, 2=down, 3=left):")
# v1.0版本修改:现在输出同一个状态下会有多个最优行为,而argmax只会选取第一个进行,所以需要修改
# print(np.reshape(np.argmax(policy, axis=1), env.shape))
update_policy_type = change_policy(policy)
print(np.reshape(update_policy_type, env.shape))
print("")

print("值函数:")
print(v)
print("")

print("值函数的网格形式:")
print(v.reshape(env.shape))
print("")

# 验证最终求出的值函数符合预期
expected_v = np.array([-4, -3, -2, -1, -2, -3, -2, -1,  0, -1, -4, -3, -2, -1, -2, -5, -4, -3, -2, -3, -6, -5, -4, -3, -4])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

