# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:23:44 2017

@author: BarryXU
"""
import pandas as pd
import numpy as np

#load data
geno_data = pd.read_table('genotype.txt', header = None)
pheno_data = pd.read_table('phenotype.txt', header = None)
geno_data = geno_data.as_matrix()
geno_data = geno_data.astype(float)
pheno_data = pheno_data.as_matrix()
# define a function to compute parameters for Guassian distribution
def gussian_para(data_array):
    return(np.mean(data_array), np.std(data_array))
def get_log_prob(data_array):
    mu, sigma = gussian_para(data_array)
    probability = list(map(lambda x : 1 / (sigma * np.sqrt(2*np.pi))* 
                                        np.exp(-(x-mu)**2 / (2*sigma**2)),
                            data_array))
    log_prob = 0
    for i in range(len(probability)):
        log_prob += np.log10(probability[i])
    return(log_prob)

# for noQTL
log_noQTL = get_log_prob(pheno_data[0,:])
# for QTL
log_byMarker = []
for row in geno_data:
    data_1 = pheno_data[0, np.where(row == 1)[0]]
    data_2 = pheno_data[0, np.where(row == 0)[0]]
    log_byMarker.append(get_log_prob(data_1) + get_log_prob(data_2))
#compute LOD and plot
LOD = log_byMarker - log_noQTL
import matplotlib.pyplot as plt
plt.hist(LOD, bins = 50)
plt.title("LOD for Markers")
plt.show()
# max LOD and its number of marker
print("-------------------------------------------------------------------------")
print("Max LOD Score: ", round(np.max(LOD), 2))
print("Number(index starts from 0) of the marker with Max LOD Score: ", np.argmax(LOD))
# 95 quantile
quantitle_95 = np.percentile(LOD, 95)
