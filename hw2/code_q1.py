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
pheno_data = pheno_data.as_matrix()[0]
#%% define function to compute parameters for Guassian distribution
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
log_noQTL = get_log_prob(pheno_data)
# for QTL
def QTL_byMarker(geno_data, pheno_data, log_noQTL, plot = False):
    log_byMarker = []
    for row in geno_data:
        data_1 = pheno_data[np.where(row == 1)[0]]
        data_2 = pheno_data[np.where(row == 0)[0]]
        log_byMarker.append(get_log_prob(data_1) + get_log_prob(data_2))
    #compute LOD and plot
    LOD = log_byMarker - log_noQTL
    if(plot):
        import matplotlib.pyplot as plt
        plt.hist(LOD, bins = 50)
        plt.title("LOD for Markers")
        plt.show()
    return (LOD)
#%% Q1
# max LOD and its number of marker
LOD = QTL_byMarker(geno_data, pheno_data, log_noQTL, plot = True)
print("-------------------------------------------------------------------------")
print("Max LOD Score: ", round(np.max(LOD), 2))
print("Number(index starts from 0) of the marker with Max LOD Score: ", round(np.argmax(LOD), 2))
#%% Q2
max_score_list = []
permute_time = 500
for i in range(int(permute_time)):
    pheno_permute = np.random.permutation(pheno_data)
    LOD_permute = QTL_byMarker(geno_data, pheno_permute, log_noQTL)
    max_score_list.append(np.max(LOD_permute))
    print(i,'/',int(permute_time))
import matplotlib.pyplot as plt
plt.hist(max_score_list, bins = 50)
plt.title("LOD for Markers")
plt.show()
quantile95 = np.percentile(max_score_list, 95)
makers_ = np.where(LOD > quantile95)










