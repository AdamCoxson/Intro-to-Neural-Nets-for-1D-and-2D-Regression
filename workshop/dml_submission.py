# -*- coding: utf-8 -*-
"""
Created: XXXX
Author: XXXXX
Department of Chemistry, Digital Chemistry Master course
Module: dml_submission.py
Local dependencies: network_classes.py, stats_and_plot_functions.py, rdkit_functions.py

Submission for the dML workshop Mar 2025 as part of the digital chemistry
masters. This applies delta learning to real world project data from the Troisi group.

See dml-brief for hints. I have done the data loading for you and given most of the 
packages you will need

"""

############## PACKAGE IMPORTS ###############
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import ast
import matplotlib.pyplot as plt

############### LOCAL IMPORTS ###############
from network_classes import MLP 
from stats_and_plot_functions import mean_squared_error, calc_rhoc, plot_loss_and_r, CustomScaler


torch.manual_seed(42) # 42 for reproducibility
num=100 # Use this for testing and debugging
#num=None set to this when you want it to load
cwd = os.getcwd()
data_train = pd.read_csv(cwd+'/data/train_data_10k_conjugated_cores.csv')[:num]
data_test = pd.read_csv(cwd+'/data/test_data_500_analogues.csv') 

print("Formatting data - this may take a moment")
# Training data
train_Etddft = data_train['S1_TDDFT'].values 
train_Ezindo = data_train['S1_ZINDO'].values
train_fp = np.array([np.array(ast.literal_eval(fp), dtype=np.int32) for fp in data_train['fingerprint'].values])
net_input= np.concatenate((train_Ezindo.reshape(-1,1), train_fp),axis=1)

# Testing data (NOT VALIDATION DATA, use this once you model has been fully trained)
test_Etddft = data_test['E(S1_TDDFT)_analogue'].values 
test_Ezindo = data_test['E(S1_ZINDO)_analogue'].values
test_fp = np.array([np.array(ast.literal_eval(fp), dtype=np.int32) for fp in data_test['fingerprint'].values])


# Initialise model
criterion = nn.MSELoss()  

    



