#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/6/12 18:15
# @Author: Vince Wu
# @File  : env.py

import numpy as np


class ChessBoardEnv:

    def __init__(self):
        self.chessboard = np.array([0, 0, 0, 0])
        self.key_dic = {0: 1, 1: 1, 14: 1, 15: 1,  # 4维立方体编码规则
                        6: 2, 7: 2, 8: 2, 9: 2,
                        4: 3, 5: 3, 10: 3, 11: 3,
                        2: 4, 3: 4, 12: 4, 13: 4}
        self.key = 0

    def reset(self):
        self.chessboard = np.array([0, 0, 0, 0])
        return self.state_to_obs()

    def step(self, action, player):
        # reward = 0
        # done = False
        # print(self.chessboard)
        if player == 1:  # 监狱长1
            assert 0 <= action <= 63
            self.key = action // 16
            action = action % 16
            for i in range(4):
                self.chessboard[3-i] = action // 2 ** (3 - i)
                action -= self.chessboard[3-i] * 2 ** (3 - i)
            return self.key, self.state_to_obs()

        elif player == 2:  # 囚犯1
            assert 0 <= action <= 3

            self.chessboard[action] = 1 - self.chessboard[action]
            return self.key, self.state_to_obs()

        elif player == 3 or player == 4:  # 监狱长2、囚犯2
            assert 0 <= action <= 5

            if action <= 2:
                self.switch(action, action + 1)
            elif action == 3:
                self.switch(3, 0)
            else:
                self.switch(action - 4, action - 2)
            if player == 4:  # 囚犯2，结束
                # print(self.state_to_key(), self.key)
                return 1 if self.state_to_key() == self.key else -1, True
            else:
                return self.key, self.state_to_obs()



    def state_to_key(self):
        key = 0
        for i in range(4):
            key += self.chessboard[i] * 2 ** i
        return self.key_dic[key]-1

    def switch(self, a, b):
        self.chessboard[a], self.chessboard[b] = self.chessboard[b], self.chessboard[a]

    def state_to_obs(self):
        obs = 0
        for i in range(4):
            obs += self.chessboard[i] * (2 ** i)
        # print(self.chessboard)
        # print(obs)
        return obs
