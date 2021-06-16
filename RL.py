#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/6/12 18:14
# @Author: Vince Wu
# @File  : RL.py

import numpy as np
import time


class Agent(object):
    def __init__(self, obs_n, act_n, player, learning_rate=0.01, gamma=0.01, e_greed=1):
        self.act_n = act_n  # 动作维度，有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))
        self.player = player

    def sample(self, obs):
        if self.epsilon <= 0.001:
            self.epsilon = 0
        else:
            self.epsilon -= (self.epsilon * 0.00005)
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # 根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # 有一定概率随机探索选取一个动作
        return action

    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        # if self.player == 2:
        #     print(obs, Q_list,action)
        return action

    def learn(self, obs, action, reward, next_obs=0, done=0):
        """ off-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            done: episode是否结束
        """

        predict_Q = self.Q[obs, action]
        target_Q = reward
        if self.Q[obs, action] == 0:
            self.Q[obs, action] = target_Q
        else:
            self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 修正q

    def save(self):
        npy_file = './q_table_{}.npy'.format(self.player)
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    def restore(self):
        self.Q = np.load('./q_table_{}.npy'.format(self.player))
        print('./q_table_{}.npy'.format(self.player) + ' loaded.')
