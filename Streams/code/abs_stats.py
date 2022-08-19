# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:36:58 2021

@author: barre
"""

import pandas as pd
import numpy as np
import os
# import datetime as dt
import matplotlib.pyplot as plt
# import scipy
# from scipy import stats
# import seaborn as sns
#from sklearn.cross_decomposition import PLSRegression
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.utils import resample
# from sklearn.metrics import mean_squared_error as MSE
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler

#for looking up available scorers
# import sklearn.metrics
# sorted(sklearn.metrics.SCORERS.keys())
# import itertools
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy import stats

#%% Set paths and bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')

abs_wq_df_fn = 'abs_wq_df_streams.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)

#%% Get abs values

abs_df = abs_wq_df.loc[:,'band_1':'band_1024']

#%% make function for regular subsetting by factor of n


def reg_sub(input_df,factor):
    
    inds = np.linspace(1,1024,num=1024,dtype = int)
    inds = inds%factor==0
    output_df = input_df.loc[:,inds]
    return(output_df)

#%% make abs_df_sub
red_fact = 6
abs_df_sub = reg_sub(abs_df,red_fact)

#%% perform pca on abs_df and abs_df_sub

pca = PCA(n_components = 20)

abs_df_red = pca.fit_transform(abs_df)
abs_df_sub_red = pca.fit_transform(abs_df_sub)

#%% plot corresponding componets

fig, axs = plt.subplots(5,4)
fig.set_size_inches(18,18)
fig.suptitle('Reduction factor = '+str(red_fact),fontsize = 28)
fig.tight_layout(h_pad = 4,w_pad = 6)
#axs[2, 2].axis('off')
row = 0
col = 0
cols = range(abs_df_red.shape[1])
for c in cols:
    abs_red = abs_df_red[:,c]
    
    abs_sub_red = abs_df_sub_red[:,c]
    
    slope, intercept, r, p, se = stats.linregress(abs_red,abs_sub_red)
    
    line11 = np.linspace(min(np.concatenate((abs_red,abs_sub_red))),
                          max(np.concatenate((abs_red,abs_sub_red))))
    
    x_line = np.array([min(abs_red),max(abs_red)])
    y_line = x_line*slope+intercept      
    
    # y_text = min(line11)+(max(line11)-min(line11))*0
    # x_text = max(line11)-(max(line11)-min(line11))*0.5
    
    y_text = min(abs_sub_red)+(max(abs_sub_red)-min(abs_sub_red))*0
    
    if slope>0:
        
        x_text = max(abs_red)-(max(abs_red)-min(abs_red))*0.5
        
    else:
        
        x_text = min(abs_red)+(max(abs_red)-min(abs_red))*0.1
    
    rsq = r2_score(abs_red,abs_sub_red)
    
    ax = axs[row,col]
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    
    axs[row,col].plot(abs_red,abs_sub_red,'o',markersize = 4)
    # axs[row,col].plot(line11,line11,'k--',label= '1:1 line')
    axs[row,col].plot(x_line,y_line,'k-',label= 'fitted line')
    # axs[row,col].set_title('component '+str(c))
    #axs[row,col].legend(loc = 'upper left',fontsize = 16)
    axs[row,col].set_xlabel('Abs. PC'+str(c),fontsize = 16,labelpad = 2)
    axs[row,col].set_ylabel('Abs. Sub. PC'+str(c),fontsize = 16,labelpad = 2)
    # axs[row,col].get_xaxis().set_visible(False)
    ax.text(x_text,y_text,r'$\/r^2 =$'+str(np.round(rsq,3)), fontsize = 16)
    # ticks = ax.get_yticks()
    # print(ticks)
    # # tick_labels = ax.get_yticklabels()
    # tick_labels =[str(round(x,1)) for x in ticks]
    # tick_labels = tick_labels[1:-1]
    # print(tick_labels)
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(tick_labels)
    
    if col == 3:
        col = 0
        row += 1
    else:
        col +=1
