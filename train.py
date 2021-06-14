#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/6/12 18:15
# @Author: Vince Wu
# @File  : train.py
from RL import Agent
from env import ChessBoardEnv
import matplotlib.pyplot as plt
import pandas as pd

def run_episode(env, agents, render=False):

    obs = []
    actions = []
    obs += [env.reset()] # 重置环境, 重新开一局（即开始新的一个episode）

    for i in range(2):
        actions += [agents[i].sample(obs[i])] # 根据算法选择一个动作
        [key, flag] = env.step(actions[i], player=i+1)  # 与环境进行一个交互
        obs += [flag]

    obs[2] += 16 * key
    actions += [agents[2].sample(obs[2])] # 根据算法选择一个动作
    [key, flag] = env.step(actions[2], player=3)  # 与环境进行一个交互
    obs += [flag]

    obs[3] += 16 * key
    actions += [agents[3].sample(obs[3])]
    reward, done = env.step(actions[3], player=4)
    # print(actions,obs)
    # 训练 Q-learning算法
    for i in range(4):
        agents[i].learn(obs[i], actions[i], reward * (i % 2 * 2 - 1))
    # print(actions,obs)
    global plot_list
    plot_list += [reward]


def test_episode(env, agent):
    obs = []
    actions = []
    obs += [env.reset()]  # 重置环境, 重新开一局（即开始新的一个episode）

    for i in range(3):
        actions += [agents[i].sample(obs[i])]  # 根据算法选择一个动作
        [key, flag] = env.step(actions[i], player=i + 1)  # 与环境进行一个交互
        obs += [flag]
    obs[3] += 16 * key
    actions += [agents[3].sample(obs[3])]
    reward, done = env.step(actions[3], player=4)

    global plot_list
    plot_list += [reward]


env = ChessBoardEnv()
agents = [Agent(1, 64, 1), Agent(16, 4, 2), Agent(64, 6, 3), Agent(64, 6, 4)]
plot_list = []
reward_list = []
# 全部训练结束，查看算法效果
for i in range(4):
    agents[i].restore()

for episode in range(200000):
    run_episode(env, agents, False)
    if episode % 1000 == 999:
        reward_list += [sum(plot_list) / 1000]
        print('episode:', episode, 'reward:', reward_list[-1])
        plot_list = []

for i in range(4):
    agents[i].save()

df = pd.DataFrame(reward_list)
writer = pd.ExcelWriter('data.xlsx', mode='a+')
df.to_excel(writer, 'sheet1')
writer.save()




plt.rcParams.update({

    "lines.color": "black",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})


plt.plot(reward_list,color='white')
plt.title('chessboard_game')
plt.xlabel('episode/1000')
plt.ylabel('reward')
plt.show()

