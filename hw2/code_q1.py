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
