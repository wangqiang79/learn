
# coding: utf-8

# In[419]:


import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.halften import HalftenEnv

# 这个函数主要用来测试环境使用
# In[420]:


env = HalftenEnv()

content = ["爆牌","平牌","十点半","五小","天王","天五小"]
# In[422]:


def print_observation(observation):
    score, card_num, p_num = observation
    print("玩家分数: {} (手牌数: {},人牌数: {})".format(
          score, card_num, p_num))

# 策略为随机策略,当前分数大于等于10时则停止叫牌
def strategy(observation):
    score, card_num,p_num = observation
    return 0 if score >= 10 else 1

for i_episode in range(20):
    observation = env._reset()
    for t in range(100):
        print_observation(observation)
        action = strategy(observation)
        print("采取的行为: {}".format( ["停牌", "叫牌"][action]))
        observation, reward, done, _ = env._step(action)
        if done:
            print_observation(observation)
            print("游戏结束,回报: {}\n".format(float(reward)))
            print("*"*50)
            break

