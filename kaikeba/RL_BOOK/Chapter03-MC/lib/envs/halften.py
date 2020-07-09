import gym
from gym import spaces
from gym.utils import seeding

# 定义牌的分数。其中，A = 1, 2-10 = 牌的点数, J/Q/K= 0.5。随机发牌就是随机的从deck中选择一张牌
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.5, 0.5, 0.5]

# 人牌值
p_val = 0.5
# 限制值
dest = 10.5

# 随机发牌，随机的从deck中选择一张牌
def draw_card(np_random):
    return np_random.choice(deck)

# 随机发到手一张牌
def draw_hand(np_random):
    return [draw_card(np_random)]

# 当前手牌总分
def sum_hand(hand):
    return sum(hand)

# 获取手牌的数量
def get_card_num(hand):
    return len(hand)

# 获取手牌中的人牌数
def get_p_num(hand):
    count = 0
    for i in hand:
        if i == p_val:
            count += 1
    return count

# 手上的牌是否爆掉
def gt_bust(hand):
    return sum_hand(hand) > dest

# 判断是否刚好达到了十点半
def is_dest(hand):
    return sum_hand(hand) == dest

# 判断是否是比十点半小
def lt_dest(hand):
    return sum_hand(hand) < dest

# 判断是否为人五小(手中牌为5张,且都为人牌)
def is_rwx(hand):
    return True if get_p_num(hand) == 5 else False

# 判断是否为天王(手中牌为5张,且牌面点数总和为十点半)
def is_tw(hand):
    return True if get_card_num(hand) == 5 and is_dest(hand) else False

# 判断是否为五小(手中牌为5张,且总点数小于十点半)
def is_wx(hand):
    return True if get_card_num(hand) == 5 and lt_dest(hand) else False

# 根据手牌返回结果(牌型,回报,结束状态)
def hand_types(hand):
    # 默认为平牌
    type = 1
    reward = 0
    done = False

    if gt_bust(hand):
        # 爆牌
        type = 0
        reward = -1
        done = True
    elif is_rwx(hand):
        # 人五小
        type = 5
        reward = 5
        done = True
    elif is_tw(hand):
        # 天王
        type = 4
        reward = 4
        done = True
    elif is_wx(hand):
        # 五小
        type = 3
        reward = 3
        done = True
    elif is_dest(hand):
        # 十点半
        type = 2
        reward = 2
        done = True
    return type,reward,done

# 庄家和玩家比较手牌
def cmp(dealer,player):
    # 规则: 庄家大,返回True,玩家大,返回False，当点数相同时比较手牌,庄家手牌数小于等于玩家,返回False大于则返回True
    dealer_score = sum_hand(dealer)
    player_score = sum_hand(player)

    if dealer_score > player_score:
        return True
    elif dealer_score < player_score:
        return False
    else:
        dealer_num = get_card_num(dealer)
        player_num = get_card_num(player)
        return True if dealer_num >= player_num else False

# 创建十点半的环境
class HalftenEnv(gym.Env):
    """
    简单十点半
    十点半是一种扑克游戏，这种游戏老少皆宜。
    游戏技巧在于如何收集成"十点半",但若超过十半点，也算失败。
　　十点半游戏中，手牌(A, 2, 3, 4, 5, 6, 7, 8, 9, 10),A为1点,其余牌点数为本身的点数,手牌（J、Q、K）为人牌,视为半点，
    现在这个环境中对局为庄家和玩家
    牌型说明:
    人五小: 5张牌,且每张都由人牌组成,奖励x5
    天王:5张牌,且牌面点数总和为十点半,奖励x4
    五小:5张牌不都是人牌,且总点数小于十点半,奖励x3
    十点半:5张牌以下,牌的总点数正好等于十点半,奖励x2
    平牌:5张牌以下,牌的总点数小于十点半,奖励x1
    爆牌:牌的总点数大于十点半
    比牌规则:
    牌型大小: 人五小>天王>五小>十点半>平牌>爆牌

    玩家拿到牌型为十点半以上(或包含)的牌(人五小,天王,五小,十点半),则立即获胜,庄家立输
    玩家拿到总分为十点半以上的牌,则为爆牌,玩家立输,庄家立即获胜

    玩家拿到十点半以下的牌并停牌,则庄家要牌,再和玩家比大小。
    庄家如果当前分数小于玩家,则继续要牌,直至分出胜负,如果庄家等于玩家分数则比较手牌的数量,若手牌数小于玩家的手牌数,则继续要牌,否则判定为庄家获胜
    庄家手牌也同样遵循牌型规则

    回报说明:
    赢牌: 1
    输牌: -1

    在计算回报时,应该根据各牌型的相应倍率进行
    """
    def __init__(self):
        # 行为空间: 停牌,叫牌
        self.action_space = spaces.Discrete(2)  # 停牌,叫牌
        # 状态空间: (玩家手牌数的总分,玩家手中的总牌数,玩家手中的人牌数)
        # 玩家的手牌总分数: 21个状态
        # 玩家的手牌数: 5个状态
        # 玩家手中的人牌数: 6个状态
        self.observation_space = spaces.Tuple((
            spaces.Discrete(21),    # 玩家当前手牌的积分
            spaces.Discrete(5),     # 手中的手牌数
            spaces.Discrete(6)))    # 手中的人牌数
        self._seed()
        # 开始牌局
        self._reset()
        # 行为数
        self.nA = 2

    # 获取随机种子
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # 基于当前的状态,和输入动作,得出下一步的状态,奖励和是否结束
    # 如果动作为叫牌: 给玩家发一张手牌,改变玩家手牌的状态。判断玩家当前手中的牌型,返回(玩家当前手牌状态,奖励和是否结束)。
    # 如果动作为停牌: 庄家开始补牌,点数比玩家大,庄家获胜,游戏结束,否则继续补牌至分出胜负(注意:当点数相同时,比较手牌,庄家手牌大于等于玩家,庄家胜,否则继续补牌)
    def _step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        # 叫牌
        if action:
            self.player.append(draw_card(self.np_random))

            # 判断当前玩家手中的牌型
            type,reward,done = hand_types(self.player)
        # 停牌
        else:
            done = True

            # 玩家停止牌之后,庄家开始补牌
            self.dealer = draw_hand(self.np_random)
            # 第一张手牌特殊牌型,所以不用判断类型,只需比较和玩家手牌大小即可
            result = cmp(self.dealer, self.player)

            if result:
                reward = -1
            else:
                while not result:
                    # 继续给庄家补牌
                    self.dealer.append(draw_card(self.np_random))

                    # 判断庄家牌型
                    dealer_type, dealer_reward, dealer_done = hand_types(self.dealer)

                    # 出现特殊牌型,终止比赛(因为上式计算的是庄家的回报,所以在转成玩家回报时应该是负值)
                    if dealer_done:
                        reward = -dealer_reward
                        break

                    # 还未终止,则对比庄家和玩家的手牌分数
                    result = cmp(self.dealer, self.player)

                    if result:
                        reward = -1
                        break


        return self._get_obs(), reward, done, {}

    # 获取当前的状态空间(玩家手牌数的总分,玩家手中的总牌数,玩家手中的人牌数)
    def _get_obs(self):
        return (sum_hand(self.player), get_card_num(self.player), get_p_num(self.player))

    # 牌局初始化
    def _reset(self):
        self.player = draw_hand(self.np_random)
        return self._get_obs()