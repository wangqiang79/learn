#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/26 15:24
# @Site    :
# @File    : __init__.py.py
# @Software: PyCharm
# @Description: 人机博弈模块

from __future__ import print_function
from game import Board, Game_UI
from mcts_alphaGoZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet # Tensorflow

class Human(object):
    """
    人类玩家
    """
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

def run():
    n = 5  # 获胜的条件(5子连成一线)
    width, height = 15, 15  # 棋盘大小(8x8)
    model_file = 'dist/best_policy.model'  # 模型文件名称
    try:
        board = Board(width=width, height=height, n_in_row=n)  # 初始化棋盘
        game = Game_UI(board, is_shown=1)  # 创建游戏对象

        # ############### 人机对弈 ###################
        # 使用TensorFlow加载训练好的模型policy_value_net
        best_policy = PolicyValueNet(width, height, model_file = model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # 人类玩家,使用鼠标落子
        human = Human()

        # 首先为人类设置start_player = 0
        game.start_play_mouse(human, mcts_player, start_player=1, is_shown=1)

        # 机器自己博弈
        # game.start_self_play(mcts_player,is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
