# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:36:12 2017

@author: Yingxin
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from sklearn.covariance import GraphLasso, GraphLassoCV, graph_lasso
from sklearn.preprocessing import normalize
import pandas as pd
import time
#%%
#Read data
sc_expression = pd.read_table("https://homes.cs.washington.edu/~suinlee/cse527/notes/yeast-comparison/scer-expression.txt", header = None)
sb_expression = pd.read_table("https://homes.cs.washington.edu/~suinlee/cse527/notes/yeast-comparison/sbay-expression.txt", header = None)
conserved_gene = pd.read_table("https://homes.cs.washington.edu/~suinlee/cse527/notes/yeast-comparison/conserved-genes.txt", header = None)
sc_experiment = pd.read_table("https://homes.cs.washington.edu/~suinlee/cse527/notes/yeast-comparison/scer-experiments.txt", header = None)
sb_experiment = pd.read_table("https://homes.cs.washington.edu/~suinlee/cse527/notes/yeast-comparison/sbay-experiments.txt", header = None)

#%%
#data normalization
sb_data = sb_expression.values.T
#sb_normdata = normalize(sb_data, axis=1)
means = np.mean(sb_data, axis=0)
stds = np.std(sb_data, axis=0)
sb_normdata = np.divide(np.subtract(sb_data, means), stds)
#%%
GL_sb = GraphLasso(alpha=1)

tic = time.time()
GL_sb.fit(sb_expression.values.T)
toc = time.time()
time1 = toc - tic
print (time1)

perc_sb = GL_sb.precision_
np.save('perc_sb.npy', perc_sb)