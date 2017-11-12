# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:53:56 2017

@author: BarryXU
"""
class Zij:
    p = 0;
    x_idx = 0;
    y_idx = 0;
    def __init__(self, p, x_idx, y_idx):
        self.p = float(p)
        self.x_idx = int(x_idx)
        self.y_idx = int(y_idx)

import numpy as np
Z = np.array([[Zij(0.5, 2, 6), Zij(0.5, 4, 5)],
              [Zij(0.5, 0, 4), Zij(0.5, 1, 3)],
              [Zij(0.5, 4, 8), Zij(0.5, 6, 7)]])
frequency_i = np.array([1/12, 1/12, 1/12, 1/12,
                      3/12, 1/12, 2/12, 1/12, 1/12])
frequency_ii = np.zeros(9)
epsilon = 1e-4
error_list = []
error = float("inf")
while(error > epsilon):
    #E-step
    for row in Z:
        tmp0 = frequency_i[row[0].x_idx] * frequency_i[row[0].y_idx]
        tmp1 = frequency_i[row[1].x_idx] * frequency_i[row[1].y_idx]
        row[0].p = tmp0 / (tmp0 + tmp1)
        row[1].p = tmp1 / (tmp0 + tmp1)
    #M-step
    for row in Z:
        for i in range(row.size):
            frequency_ii[row[i].x_idx] += row[i].p / Z.size
            frequency_ii[row[i].y_idx] += row[i].p / Z.size
    error = np.sum(abs(frequency_i - frequency_ii))
    error_list.append(error)
    frequency_i = frequency_ii
