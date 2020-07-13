import numpy as np
 
world_h = 4
world_w = 4
length = world_h * world_w
gamma = 1
state = [i for i in range(length)]
action = ['n', 'e', 's', 'w']
ds_action = {'n':-world_w, 'e':1, 's':world_w, 'w':-1}
value = [0 for i in range(length)]

#定义奖励
def reward(s):
    return 0 if s in [0, length - 1] else -1
 
def next_states(s, a):
    next_state = s
    if (s < world_w  and a == 'n') or (s % world_w == 0 and a == 'w')\
        or (s > length - world_w - 1 and a == 's') or ((s+1) % world_w == 0 and a == 'e'):  #(s % (world_w - 1) == 0 and a == 'e' and s != 0)
        pass
    else:
       next_state = s + ds_action[a]
    return next_state
 
def getsuccessor(s):
    successor = []
    for a in action:
        next = next_states(s, a)
        successor.append(next)
    return successor
#更新V值
def value_update(s):
    value_new = 0
    if s in [0, length - 1]:
        pass
    else:
        successor = getsuccessor(s)
        rewards = reward(s)
        for next_state in successor:
            value_new += 0.25*(rewards + gamma * value[next_state])   #1.00 / len(action)
 
    return value_new
 
def main():
    max_iter = 201
    global value
    v = np.array(value).reshape(world_h, world_w)
    print(v)
    iter = 1
    while iter < max_iter:
        new_value = [0 for i in range(length)]
        for s in state:
            new_value[s] = value_update(s)
        value = new_value
        v = np.array(value).reshape(world_h, world_w)
        if (iter<=10) or (iter%10 == 0):
           print('k=',iter)
           print(np.round(v,decimals=2))
        iter += 1
 
if __name__ == '__main__':
    main()

