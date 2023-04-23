# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:39:55 2021

@author: jbarrett.carter
"""
#%% import libraries

import pandas as pd
import numpy as np
import os
# import datetime as dt
import matplotlib.pyplot as plt
# import scipy
# from scipy import stats
# import seaborn as sns
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.utils import resample
# from sklearn.metrics import mean_squared_error as MSE
# import xgboost as xgb


#for looking up available scorers
import sklearn.metrics
sorted(sklearn.metrics.SCORERS.keys())

# from joblib import dump

#%% Set paths and bring in data

user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\water_quality-spectroscopy' #for laptop
# path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')

outputs_df_fn = 'streams_XGB_It1_md1-20_lr001-02_results.csv'

# Bring in data
outputs_df=pd.read_csv(output_dir+outputs_df_fn)

#%% Define function for making plots

s = 'Nitrate-N'

def make_plots(outputs_df, output_label):

    ## make plots for both filtered and unfiltered samples
        
    fig, axs = plt.subplots(3,3)
    fig.set_size_inches(15,15)
    fig.suptitle(output_label,fontsize = 18)
    fig.tight_layout(pad = 4)
    axs[2, 2].axis('off')
    row = 0
    col = 0
    species = outputs_df.species.unique()
    for s in species:
        y_true_train = outputs_df.loc[((outputs_df.species == s) &
                                        (outputs_df.output == 'y_true_train')),
                                       'value']
        
        y_hat_train = outputs_df.loc[((outputs_df.species == s) &
                                        (outputs_df.output == 'y_hat_train')),
                                       'value']
        
        y_true_test = outputs_df.loc[((outputs_df.species == s) &
                                        (outputs_df.output == 'y_true_test')),
                                       'value']
        
        y_hat_test = outputs_df.loc[((outputs_df.species == s) &
                                        (outputs_df.output == 'y_hat_test')),
                                       'value']
        
        line11 = np.linspace(min(np.concatenate((y_true_train,y_hat_train,
                                                 y_true_test,y_hat_test))),
                              max(np.concatenate((y_true_train,y_hat_train,
                                                 y_true_test,y_hat_test))))
        
        y_text = min(line11)+(max(line11)-min(line11))*0
        x_text = max(line11)-(max(line11)-min(line11))*0.5
        
        train_rmse = outputs_df['value'][(outputs_df.output == 'train_rmse')&
                            (outputs_df.species==s)]
        
        train_rmse = np.mean(train_rmse)
        
        test_rmse = outputs_df['value'][(outputs_df.output == 'test_rmse')&
                            (outputs_df.species==s)]
        
        test_rmse = np.mean(test_rmse)
        
        learning_rate = outputs_df['value'][(outputs_df.output == 'learning_rate')&
                            (outputs_df.species==s)]
        
        learning_rate = np.mean(learning_rate)
        
        max_depth = outputs_df['value'][(outputs_df.output == 'max_depth')&
                            (outputs_df.species==s)]
        
        ax = axs[row,col]
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)
        
        axs[row,col].plot(y_true_train,y_hat_train,'o',markersize = 4, label = 'training set')
        axs[row,col].plot(y_true_test,y_hat_test,'o',markersize = 4, label = 'test set')
        axs[row,col].plot(line11,line11,'k--',label= '1:1 line')
        # axs[row,col].set_title(s)
        if (row == 0) and (col == 0):
            axs[row,col].legend(loc = 'upper left',fontsize = 16)
        axs[row,col].set_xlabel('Lab Measured '+s+' (mg/L)',fontsize = 16)
        axs[row,col].set_ylabel('Predicted '+s+' (mg/L)',fontsize = 16)
        # axs[row,col].get_xaxis().set_visible(False)
        ax.text(x_text,y_text,'$rmse_{tr} =$'+str(np.round(train_rmse,3))+'\n'
                +'$rmse_{te} =$'+str(np.round(test_rmse,3))+'\n'
                +'$lr =$'+str(np.round(learning_rate,2))+'\n'
                +'$md =$'+str(int(max_depth)), fontsize = 16)
        # ticks = ax.get_yticks()
        # print(ticks)
        # # tick_labels = ax.get_yticklabels()
        # tick_labels =[str(round(x,1)) for x in ticks]
        # tick_labels = tick_labels[1:-1]
        # print(tick_labels)
        # ax.set_xticks(ticks)
        # ax.set_xticklabels(tick_labels)
        
        if col == 2:
            col = 0
            row += 1
        else:
            col +=1
    # fig.show()

#%% make plots

make_plots(outputs_df, 'XGB: Learning Rate = 0.01 - 0.2; Max Depth = 1 - 20')
