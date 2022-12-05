# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:09:45 2022

@author: jbarrett.carter
"""

#%% import libraries

from joblib import load
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import scipy
# from scipy import stats
# import seaborn as sns

#%% Set paths and bring in models

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')

abs_wq_df_fn = 'abs_wq_df_streams.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)

filename = 'pls_Nitrate-N_It0.joblib'
pickle_path = os.path.join(output_dir,'picklejar',filename)

#bring in model
clf = load(pickle_path)

#%% Test model

input_df = abs_wq_df
s = 'Nitrate-N'
iteration = 1

Y = input_df[s]
keep = pd.notna(Y)
X = input_df.loc[keep,'band_1':'band_1024']
Y = Y[keep]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=iteration,
                                                                test_size = 0.3)

mod = clf.best_estimator_

y_hat = mod.predict(X_test)

plt.scatter(y_hat,y_test)
plt.ylabel('True')
plt.xlabel('Predicted')