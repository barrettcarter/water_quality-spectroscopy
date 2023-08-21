# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 07:41:34 2023

The purpose of this script is to determine the sample size for each species in
the streams samples so that the filtered and unfiltered subsets have the same
sample size.

@author: carter_j
"""

#%% import libraries

import pandas as pd
import numpy as np
import os
# import datetime as dt
# import matplotlib.pyplot as plt
# import scipy
# from scipy import stats
# import seaborn as sns
# from sklearn.cross_decomposition import PLSRegression
# # from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# # from sklearn.linear_model import LinearRegression
# # from sklearn.utils import resample
# from sklearn.metrics import mean_squared_error as MSE
# # from sklearn.ensemble import RandomForestRegressor

# #for looking up available scorers
# # import sklearn.metrics
# # sorted(sklearn.metrics.SCORERS.keys())

# from joblib import dump

#%% Set paths and bring in data

user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\' for OneDrive
# path_to_wqs = '/blue/ezbean/jbarrett.carter/water_quality-spectroscopy/' # for HiPerGator
path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')

abs_wq_df_fn = 'abs_wq_df_streams.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)

#%% seperate into filtered and unfiltered sample sets and make useful variables

abs_wq_df_fil = abs_wq_df.loc[abs_wq_df['Filtered']==True,:]
abs_wq_df_unf = abs_wq_df.loc[abs_wq_df['Filtered']==False,:]

subset_name_fil = 'streams_fil'
subset_name_unf = 'streams_unf'

species = abs_wq_df.columns[0:8]

#%% determine sample size for each species based on minimum n of filtered and unfiltered subsets

sam_size_df = pd.DataFrame(columns = ['Species','Samp_size'])

s = species[2]

for s in species:
    
    df_s_fil = abs_wq_df_fil.loc[abs_wq_df_fil[s].isna()==False,s]
    df_s_unf = abs_wq_df_unf.loc[abs_wq_df_unf[s].isna()==False,s]
    
    n_s = min([df_s_fil.shape[0],df_s_unf.shape[0]])

    new_row = pd.DataFrame(columns = ['Species','Samp_size'])
    
    new_row.loc[0,:] = [s,n_s]
    
    sam_size_df = pd.concat([sam_size_df,new_row],ignore_index=True)
    
#%% save output

sam_size_df.to_csv(os.path.join(inter_dir,'fil_sub_samp_sizes.csv'),index=False)
