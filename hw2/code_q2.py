# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:50:00 2017

@author: BarryXU
"""
import pandas as pd
import numpy as np
#load data
geno_data = pd.read_table('genotype.txt', header = None)
pheno_data = pd.read_table('phenotype.txt', header = None)
geno_data = geno_data.as_matrix().T
geno_data = geno_data.astype(float)
pheno_data = pheno_data.as_matrix().T
#split data and normalizaiton
from sklearn.preprocessing import scale
geno_norm = scale(geno_data) 
train_geno = geno_norm[0:250, :]
test_geno = geno_norm[250:, :]
train_pheno = pheno_data[0:250, :]
test_pheno = pheno_data[250:, :]
#%%
def pred_model(train_X, test_X, train_y, test_y, alpha_range, model = "lasso", plot = True):
    from sklearn import linear_model
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    mse = []
    for i in alpha_range:
        if(model == "lasso"):
            clf = linear_model.Lasso(alpha = i)
        elif(model == "ridge"):
            clf = linear_model.Ridge(alpha = i)            
        clf.fit(train_X, train_y)
        pred = clf.predict(test_X)
        mse.append(mean_squared_error(pred, test_y))
    # plot
    if(plot):
        plt.plot(alpha_range, mse)
        if(model == "lasso"):
            plt.title("Lasso")
        elif(model == "ridge"):
            plt.title("Ridge")
        plt.xlabel("Alpha Value")
        plt.ylabel("Mean Square Error")
        plt.show()
    return mse
#lasso regression
lasso_mse = pred_model(train_geno, test_geno, train_pheno, test_pheno, 
                       np.arange(0.01, 0.3, 0.01))
min(lasso_mse)
#ridge regression
ridge_mse = pred_model(train_geno, test_geno, train_pheno, test_pheno, 
                       np.arange(1000, 2000, 50),model = "ridge")
min(ridge_mse)
#%% leave one out validation
#trainset : valset = 2:1
from sklearn.model_selection import train_test_split
X_geno, V_geno, X_pheno, V_pheno = train_test_split(train_geno, train_pheno, 
                                                    test_size=0.33, random_state=55)
#lasso
lasso_mse_val = pred_model(X_geno, V_geno, X_pheno, V_pheno,
                           np.arange(0.05, 0.1, 0.001))
#alpha_lasso = 0.076
#train + val = new train
lasso_mse_test = pred_model(train_geno, test_geno, train_pheno, test_pheno, 
                            [0.076], plot = False)

#ridge
ridge_mse_val = pred_model(X_geno, V_geno, X_pheno, V_pheno,
                           np.arange(2e4, 3e4, 200), model = "ridge")
#alpha_ridge = 24200
ridge_mse_test = pred_model(train_geno, test_geno, train_pheno, test_pheno, 
                            [24200], model = "ridge", plot = False)














