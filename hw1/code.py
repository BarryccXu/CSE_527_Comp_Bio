"""
Created on Wed Oct 18 00:13:52 2017
@author: BarryXU
"""
import numpy as np
import pandas as pd
#%%
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
#%% model 2
M2_theta_80 = None
M2_theta_4 = None
M2_theta_2 = [None] * 4
M2_theta_80 = len(ones_theta_80[0]) / len(data[0, :]) 
M2_theta_4 = len(ones_theta_4[0]) / len(data[1, :])
tmp = np.intersect1d(zeros_theta_80[0], zeros_theta_4[0])
M2_theta_2[0] = len(np.intersect1d(tmp, ones_theta_2[0])) / len(tmp)
tmp = np.intersect1d(zeros_theta_80[0], ones_theta_4[0])
M2_theta_2[1] = len(np.intersect1d(tmp, ones_theta_2[0])) / len(tmp)
tmp = np.intersect1d(ones_theta_80[0], zeros_theta_4[0])
M2_theta_2[2] = len(np.intersect1d(tmp, ones_theta_2[0])) / len(tmp)
tmp = np.intersect1d(ones_theta_80[0], ones_theta_4[0])
M2_theta_2[3] = len(np.intersect1d(tmp, ones_theta_2[0])) / len(tmp)
