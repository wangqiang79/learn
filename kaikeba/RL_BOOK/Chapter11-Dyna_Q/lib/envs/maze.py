
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # 每个格子的大小
MAZE_H = 5  # 行数
MAZE_W = 5  # 列数

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.nS = np.prod([MAZE_H , MAZE_W])
        self.n_actions = len(self.action_space)
        self.title('寻宝')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        # 创建一个画布
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # 在画布上画出列
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        # 在画布上画出行
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 创建探险者起始位置(默认为左上角)
        origin = np.array([20, 20])

        # 陷阱1
        hell1_center = origin + np.array([UNIT, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # 陷阱2
        hell2_center = origin + np.array([UNIT*2, UNIT])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # 陷阱3
        hell3_center = origin + np.array([UNIT*3, UNIT])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')

        # 陷阱4
        hell4_center = origin + np.array([UNIT, UNIT * 3])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')

        # 陷阱5
        hell5_center = origin + np.array([UNIT*3, UNIT * 3])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')

        # 陷阱6
        hell6_center = origin + np.array([0, UNIT * 4])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='black')

        # 陷阱7
        hell7_center = origin + np.array([UNIT*4, UNIT * 4])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 15, hell7_center[1] - 15,
            hell7_center[0] + 15, hell7_center[1] + 15,
            fill='black')

        # 宝藏位置
        oval_center = origin + np.array([UNIT * 2,UNIT*4])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # 将探险者用矩形表示
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # 画布展示
        self.canvas.pack()

    # 根据当前的状态重置画布(为了展示动态效果)
    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        return self.canvas.coords(self.rect)

    # 根据当前行为,确认下一步的位置
    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # 上
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # 下
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # 左
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:   # 右
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT

        # 在画布上将探险者移动到下一位置
        self.canvas.move(self.rect, base_action[0], base_action[1])
        # 重新渲染整个界面
        s_ = self.canvas.coords(self.rect)
        oval_flag = False

        # 根据当前位置来获得回报值,及是否终止
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
            oval_flag = True
        elif s_ in [self.canvas.coords(self.hell1),self.canvas.coords(self.hell2), self.canvas.coords(self.hell3),self.canvas.coords(self.hell4), self.canvas.coords(self.hell5),self.canvas.coords(self.hell6), self.canvas.coords(self.hell7)]:
            reward = -1
            done = False
            # done = True
            # s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done, oval_flag

    def render(self):
        time.sleep(0.1)
        self.update()

    # 根据传入策略进行界面的渲染
    def render_by_policy(self,policy):
        cal_policy = sorted(policy)

        pre_x, pre_y = 20, 20

        for state in cal_policy:
            x = (state[0] + state[2]) / 2
            y = (state[1] + state[3]) / 2

            self.canvas.create_line(pre_x, pre_y, x, y,fill = "red",tags = "line",width=5)

            pre_x = x
            pre_y = y

        # 连接到宝藏位置
        oval_center = [20,20] + np.array([UNIT * 2, UNIT * 4])

        self.canvas.create_line(pre_x, pre_y, oval_center[0], oval_center[1], fill="red", tags="line", width=5)

        self.render()

    def render_by_policy_new(self,policy):
        for i in range(MAZE_W):
            rows_obj = policy[i]
            for j in range(MAZE_H):
                item_center_x, item_center_y = (j * UNIT + UNIT/2), (i * UNIT + UNIT/2)

                cols_obj = rows_obj[j]

                if cols_obj == -1:
                    continue

                for item in cols_obj:
                    if item == 0:
                        item_x = item_center_x
                        item_y = item_center_y - 15.0
                        self.canvas.create_line(item_center_x, item_center_y, item_x, item_y, fill="black", width=1,arrow='last')
                    elif item == 1:
                        item_x = item_center_x
                        item_y = item_center_y + 15.0
                        self.canvas.create_line(item_center_x, item_center_y, item_x, item_y, fill="black", width=1,
                                                arrow='last')
                    elif item == 2:
                        item_x = item_center_x - 15.0
                        item_y = item_center_y
                        self.canvas.create_line(item_center_x, item_center_y, item_x, item_y, fill="black", width=1,
                                                arrow='last')
                    elif item == 3:
                        item_x = item_center_x + 15.0
                        item_y = item_center_y
                        self.canvas.create_line(item_center_x, item_center_y, item_x, item_y, fill="black", width=1,
                                                arrow='last')

        self.render()
