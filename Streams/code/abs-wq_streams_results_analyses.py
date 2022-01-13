# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 08:50:22 2021

@author: jbarrett.carter
"""

import pandas as pd

import os
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
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
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error as MSE
import seaborn as sns

#%% Set parameters

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
output_dir = os.path.join(path_to_wqs,'Streams/outputs/')
results_pls_fn = 'streams_PLS_It0-9_results.csv'
results_dl_fn = 'streams_DL_It0-9_results.csv'
results_path = os.path.join(path_to_wqs,output_dir)

# sns.set_style("ticks")
# sns.set_palette('colorblind')
sns.set(style = 'ticks',font_scale=2, palette = 'colorblind')

#%% Bring in data

results_pls_df=pd.read_csv(os.path.join(results_path,results_pls_fn))
results_dl_df=pd.read_csv(os.path.join(results_path,results_dl_fn))

#%% combine data frames

# add model columns
results_pls_df['model'] = 'pls'
results_dl_df['model'] = 'dl'
results_df = pd.concat([results_pls_df,results_dl_df],ignore_index=True)
results_df.loc[:,'value']=results_df.loc[:,'value'].apply(lambda x: float(x))

#%% make some useful variables

species = results_df['species'].unique()
species = list(species)
species.sort(key = lambda x: x[-1])

iterations = results_df.iteration.unique()

models = results_df.model.unique()

#%% get rmse values

test_rmses = results_df.loc[results_df['output'] == 'test_rmse',:]
test_rmses.reset_index(inplace = True)
# test_rmses.value = test_rmses.value.apply(lambda x: float(x))
test_rmses.rename(columns = {'value':'test rmse (ppm)'},inplace = True)

#%% make rmse violin plot plot

test_rmse_plot = sns.catplot(x='model',y='test rmse (ppm)',col = 'species',col_wrap=3,
                             data = test_rmses,kind = 'violin')

#%% data wrangling for 1:1 plots

y_hat_tests = results_df.loc[results_df.output=='y_hat_test',:].reset_index(drop=True)
y_true_tests = results_df.loc[results_df.output=='y_true_test',:].reset_index(drop=True)

#%% make dataframe for plot

fit_plot_df = pd.DataFrame(columns = ['True Concentration','Predicted Concentration','Model','Species'])
fit_plot_df['True Concentration']=y_true_tests.value
fit_plot_df['Predicted Concentration']=y_hat_tests.value
fit_plot_df['Model']=y_hat_tests.model
fit_plot_df['Species']=y_hat_tests.species

#%% make plot

sns.relplot(data = fit_plot_df, x = 'Predicted Concentration',y = 'True Concentration',
            hue = 'Model', col = 'Species',col_wrap = 3)

#%% make normalized rmses

rmses_norm_test = test_rmses.copy()
rmses_norm_test.rename(columns = {'test rmse (ppm)':'value'},inplace = True)

# s, i, m = species[0], iterations [0], models[0]

for s in species:
    for i in iterations:
        for m in models:
            
            row = (rmses_norm_test.species == s) \
                & (rmses_norm_test.iteration == i) \
                    & (rmses_norm_test.model == m)
            
            rows = (y_true_tests.species == s) \
                & (y_true_tests.iteration == i) \
                    & (y_true_tests.model == m)
            
            rmse = rmses_norm_test.loc[row,'value']
            av = y_true_tests.loc[rows,'value'].mean()
            rmses_norm_test.loc[row,'value'] = rmse/av
            


#%% save average normalized test rmses

rmses_norm_test_av = pd.DataFrame(columns= ['model','species','value'])

s, i, m = species[0], iterations [0], models[0]

for s in species:
    for m in models:
        
        rows = (rmses_norm_test.species == s) & (rmses_norm_test.model == m)
            
        rmse_norm_av = rmses_norm_test.loc[rows,'value'].mean()
        
        new_row = pd.DataFrame([[m,s,rmse_norm_av]],columns=['model','species','value'])
        
        rmses_norm_test_av = rmses_norm_test_av.append(new_row)

#%% calculate overall average normalized rmse

rmse_av_norm = rmses_norm_test_av.value.mean()
                
#%% make plot of normalized rmses

rmses_norm_test.rename(columns = {'value':'normalized test rmse'},inplace = True)
norm_test_rmse_plot = sns.catplot(x='model',y='normalized test rmse',
                                  col = 'species',col_wrap=3,
                             data = rmses_norm_test,kind = 'violin')
rmses_norm_test.rename(columns = {'normalized test rmse (ppm)':'value'},inplace = True)  

#%% calculate NSEs

NSEs_test = test_rmses.copy()
NSEs_test.rename(columns = {'test rmse (ppm)':'value'},inplace = True)

s, i, m = species[0], iterations [0], models[0]

for s in species:
    for i in iterations:
        for m in models:
            
            row = (NSEs_test.species == s) \
                & (NSEs_test.iteration == i) \
                    & (NSEs_test.model == m)
            
            rows = (y_true_tests.species == s) \
                & (y_true_tests.iteration == i) \
                    & (y_true_tests.model == m)
            
            y_hats = y_hat_tests.loc[rows,'value']
            y_trues = y_true_tests.loc[rows,'value']
            
            errors = (y_hats-y_trues)
            sq_errors = errors*errors
            
            sse = sq_errors.sum()
            
            av = y_true_tests.loc[rows,'value'].mean()
            
            errors = (av-y_trues)
            sq_errors = errors*errors
            
            sse_av = sq_errors.sum()
            
            nse = 1 - sse/sse_av

            NSEs_test.loc[row,'value'] = nse

#%% make plot of NSEs without ammonium (too large of range)

NSEs_test.rename(columns = {'value':'NSE'},inplace = True)
NSEs_test_plot = sns.catplot(x='model',y='NSE',
                                  col = 'species',col_wrap=3,
                             data = NSEs_test.loc[NSEs_test.species != 'Ammonium-N',:],
                             kind = 'violin')
NSEs_test.rename(columns = {'NSE':'value'},inplace = True)  

#%% save average NSEs

NSEs_test_av = pd.DataFrame(columns= ['model','species','value'])

# s, i, m = species[0], iterations [0], models[0]

for s in species:
    for m in models:
        
        rows = (NSEs_test.species == s) & (NSEs_test.model == m)
            
        NSE_av = NSEs_test.loc[rows,'value'].mean()
        
        new_row = pd.DataFrame([[m,s,NSE_av]],columns=['model','species','value'])
        
        NSEs_test_av = NSEs_test_av.append(new_row)

#%% calculate overall average normalized rmse

NSE_test_av = NSEs_test_av.value.mean()
