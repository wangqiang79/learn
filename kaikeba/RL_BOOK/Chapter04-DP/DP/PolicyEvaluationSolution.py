# coding: utf-8

import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

#  policy_eval方法是策略评估方法，输入要评估的策略policy_eval，环境env，折扣因子，阈值。输出当前策略下收敛的值函数v
def policy_eval(policy, env, discount_factor=1, threshold=0.00001):  # discount_factor=0.9
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        threshold: We stop evaluation once our value function change is less than threshold for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # 初始化各状态的状态值函数
    V = np.zeros(env.nS)
    i = 0
    print("第%d次输出各个状态值为" % i)
    print(V.reshape(5,5))
    print("-"*50)
    while True:
        value_delta = 0
        # 遍历各状态
        for s in range(env.nS):
            v = 0
            # 遍历各行为的概率(上,右,下,左)
            for a, action_prob in enumerate(policy[s]):
                # 对于每个行为确认下个状态
                # 四个参数: prob：概率, next_state: 下一个状态的索引, reward: 回报, done: 是否是终止状态
                for prob, next_state, reward, done in env.P[s][a]:
                    # 使用贝尔曼期望方程进行状态值函数的求解
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # 求出各状态和上一次求得状态的最大差值
            value_delta = max(value_delta, np.abs(v - V[s]))
            V[s] = v
        i += 1
        print("第%d次输出各个状态值为"%i)
        print(V.reshape(5,5))
        print("-" * 50)
        # 当前循环得出的各状态和上一次状态的最大差值小于阈值,则收敛停止运算
        if value_delta < threshold:
            print("第%d后,所得结果已经收敛,运算结束"%i)
            break
    return np.array(V)


env = GridworldEnv()
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

print("最终值函数:")
print(v)

print("值函数的网格形式:")
print(v.reshape(env.shape))

# 验证最终求出的状态值函数符合预期
expected_v = np.array([-47,-42,-31,-18,-20,-48,-42,-29,0,-18,-51,-47,-39,-29,-31,-54,-52,-47,-43,-42,-57,-55,-52,-48,-47])
np.testing.assert_array_almost_equal(v, expected_v, decimal=0)

