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
import statsmodels.api as sm
from matplotlib.axis import Tick
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

filename = 'RF_Nitrate-N_It0.joblib'
pickle_path = os.path.join(output_dir,'picklejar',filename)

#bring in model
clf = load(pickle_path)

#%% Test model

input_df = abs_wq_df
s = 'Nitrate-N'
iteration = 0

Y = input_df[s]
keep = pd.notna(Y)
X = input_df.loc[keep,'band_1':'band_1024']
Y = Y[keep]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=iteration,
                                                                test_size = 0.3)

mod = clf.best_estimator_

y_hat = mod.predict(X_test)

plt.scatter(y_test,y_hat)
plt.ylabel('Predicted')
plt.xlabel('True')

#%% Look at model tuning parameters

tuning_results = clf.cv_results_
scores = np.array(tuning_results['mean_test_score']*-1)
alphas = np.array(tuning_results['param_ccp_alpha'])
max_feats = np.array(tuning_results['param_max_features'])
max_feats = max_feats*1024

#%% Make plots

fig, axs = plt.subplots(1,2)
axs[0].plot(alphas,scores,'o')
axs[0].set_xlabel('pruning alpha values')
axs[0].set_ylabel('SSE')

axs[1].plot(max_feats,scores,'o')
axs[1].set_xlabel('number of wavelengths')
w = axs[1].get_yaxis() 
     
Tick.set_visible(w, False)

#plt.scatter(alphas,scores)

#%% test for significance

X_lm = pd.DataFrame({'alphas':alphas,'max_feats':max_feats})
X_lm['alphas']=X_lm['alphas'].apply(lambda x: float(x))
X_lm['max_feats']=X_lm['max_feats'].apply(lambda x: float(x))
X_lm = sm.add_constant(X_lm)
#X_lm['axm']=X_lm.alphas*X_lm.max_feats

y = pd.DataFrame({'scores':scores})

lr_1 = sm.OLS(y, X_lm).fit()

print(lr_1.summary())

#%%
plt.scatter(alphas*max_feats,scores)
