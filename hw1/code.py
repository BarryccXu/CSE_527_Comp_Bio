"""
Created on Wed Oct 18 00:13:52 2017
@author: Chenchao Xu
"""
import numpy as np
import pandas as pd
#%% load data, count zeros and ones
gene = pd.read_table('disc-gal80-gal4-gal2.txt')
name = gene['EXP'].tolist()
data = gene.ix[:,1:].as_matrix()
ones_theta_80 = np.where(data[0, :] == 1)
zeros_theta_80 = np.where(data[0, :] == 0)
ones_theta_4 = np.where(data[1, :] == 1)
zeros_theta_4 = np.where(data[1, :] == 0)
ones_theta_2 = np.where(data[2, :] == 1)
zeros_theta_2 = np.where(data[2, :] == 0)
#%% model 1
M1_theta_80 = None
M1_theta_4 = [None] * 2
M1_theta_2 = [None] * 2
M1_theta_80 = len(ones_theta_80[0]) / len(data[0, :]) 
M1_theta_4[0] = len(np.intersect1d(zeros_theta_80[0], ones_theta_4[0])) / len(zeros_theta_80[0]) 
M1_theta_4[1] = len(np.intersect1d(ones_theta_80[0], ones_theta_4[0])) / len(ones_theta_80[0])
M1_theta_2[0] = len(np.intersect1d(zeros_theta_4[0], ones_theta_2[0])) / len(zeros_theta_4[0])
M1_theta_2[1] = len(np.intersect1d(ones_theta_4[0], ones_theta_2[0])) / len(ones_theta_4[0])
'''
print('Parameters of Model 1:')
print('M1_theta_80: ', round(M1_theta_80, 3))
'''
#%% model 2
M2_theta_80 = None
M2_theta_4 = None
M2_theta_2 = [None] * 4
M2_theta_80 = len(ones_theta_80[0]) / len(data[0, :]) 
M2_theta_4 = len(ones_theta_4[0]) / len(data[1, :])
tmp = [None] * 4
tmp[0] = np.intersect1d(zeros_theta_80[0], zeros_theta_4[0])
M2_theta_2[0] = len(np.intersect1d(tmp[0], ones_theta_2[0])) / len(tmp[0])
tmp[1] = np.intersect1d(zeros_theta_80[0], ones_theta_4[0])
M2_theta_2[1] = len(np.intersect1d(tmp[1], ones_theta_2[0])) / len(tmp[1])
tmp[2] = np.intersect1d(ones_theta_80[0], zeros_theta_4[0])
M2_theta_2[2] = len(np.intersect1d(tmp[2], ones_theta_2[0])) / len(tmp[2])
tmp[3] = np.intersect1d(ones_theta_80[0], ones_theta_4[0])
M2_theta_2[3] = len(np.intersect1d(tmp[3], ones_theta_2[0])) / len(tmp[3])
#%% evaluation
# calculate lob based likehood
def cal_likehood(ones, zeros, theta):
    likehood = len(ones) * np.log(theta) + len(zeros) * np.log(1 - theta)
    return likehood
#model 1
M1_likehood = cal_likehood(ones_theta_80, zeros_theta_80, M1_theta_80) \
            + cal_likehood(np.intersect1d(zeros_theta_80[0], ones_theta_4[0]), np.intersect1d(zeros_theta_80[0], zeros_theta_4[0]), M1_theta_4[0]) \
            + cal_likehood(np.intersect1d(ones_theta_80[0], ones_theta_4[0]), np.intersect1d(ones_theta_80[0], zeros_theta_4[0]), M1_theta_4[1]) \
            + cal_likehood(np.intersect1d(zeros_theta_4[0], ones_theta_2[0]), np.intersect1d(zeros_theta_4[0], zeros_theta_2[0]), M1_theta_2[0]) \
            + cal_likehood(np.intersect1d(ones_theta_4[0], ones_theta_2[0]), np.intersect1d(ones_theta_4[0], zeros_theta_2[0]), M1_theta_2[1]) 

# model 2
M2_likehood = cal_likehood(ones_theta_80, zeros_theta_80, M2_theta_80) \
            + cal_likehood(ones_theta_4, zeros_theta_4, M2_theta_4) \
            + cal_likehood(np.intersect1d(tmp[0], ones_theta_2[0]), np.intersect1d(tmp[0], zeros_theta_2[0]), M2_theta_2[0])\
            + cal_likehood(np.intersect1d(tmp[1], ones_theta_2[0]), np.intersect1d(tmp[1], zeros_theta_2[0]), M2_theta_2[1])\
            + cal_likehood(np.intersect1d(tmp[2], ones_theta_2[0]), np.intersect1d(tmp[2], zeros_theta_2[0]), M2_theta_2[2])\
            + cal_likehood(np.intersect1d(tmp[3], ones_theta_2[0]), np.intersect1d(tmp[3], zeros_theta_2[0]), M2_theta_2[3])








