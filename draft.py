#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/6/12 18:33
# @Author: Vince Wu
# @File  : draft.py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({

    "lines.color": "black",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "white",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})
for key in plt.rcParams.keys():
    if plt.rcParams[key] == "blue":
        plt.rcParams[key] = "white"
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
x = range(1,25)
y = range(60,108)[::-2]
y2 = range(16,40)[::-1]

plt.plot(x, y,color='white')


plt.xlabel('Hour of day')
plt.ylabel('Value')
plt.title('Experimental results', color="w")

plt.tight_layout()
plt.show()
