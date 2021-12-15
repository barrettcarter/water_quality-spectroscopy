# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 08:50:22 2021

@author: jbarrett.carter
"""

import pandas as pd

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from keras.models import Sequential
# # from keras.layers import Conv1D, MaxPool1D, AveragePooling1D, Dense, Flatten, Dropout
# from keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, Dropout
# from keras.regularizers import l2
# # from keras.constraints import maxnorm
# from keras.optimizers import Adam
# import skopt
# from scipy.signal import savgol_filter
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
import seaborn as sns

#%% Set parameters

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
output_dir = os.path.join(path_to_wqs,'Streams/outputs/')
results_pls_fn = 'streams_PLS_It10_results.csv'
results_dl_fn = 'streams_DL_It0-9_results.csv'
results_path = os.path.join(path_to_wqs,output_dir)

sns.set_style("ticks")
sns.set_palette('colorblind')

#%% Bring in data

results_pls_df=pd.read_csv(os.path.join(results_path,results_pls_fn))
results_dl_df=pd.read_csv(os.path.join(results_path,results_dl_fn))




#%% combine data frames

# add model columns
results_pls_df['model'] = 'pls'
results_dl_df['model'] = 'dl'
results_df = pd.concat([results_pls_df,results_dl_df],ignore_index=True)

species = results_df['species'].unique()
species = list(species)
species.sort(key = lambda x: x[-1])

#%% get rmse values

test_rmses = results_df.loc[results_df['output'] == 'test_rmse',:]
test_rmses.reset_index(inplace = True)
test_rmses.value = test_rmses.value.apply(lambda x: float(x))

test_rmse_plot = sns.catplot(x='model',y='value',col = 'species',col_wrap=2,
                             data = test_rmses,kind = 'violin')
