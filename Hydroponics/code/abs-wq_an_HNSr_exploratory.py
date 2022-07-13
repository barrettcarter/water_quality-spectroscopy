# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:29:57 2022

This script is for performing statistical characterization of the real
hydroponic data. One goal of this code is to remove data 

@author: jbarrett.carter
"""

import pandas as pd
import os
#import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

#%% Set paths and bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
inter_dir=os.path.join(path_to_wqs,'Hydroponics/intermediates/')
output_dir=os.path.join(path_to_wqs,'Hydroponics/outputs/')

abs_wq_df_fn = 'abs-wq_HNSr_df.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)
abs_wq_df = abs_wq_df.loc[0:62,:]

#%% Useful variables

species = abs_wq_df.columns[0:14]

#%% make histograms

def make_plots(outputs_df, output_label):
        
    fig, axs = plt.subplots(5,3)
    fig.set_size_inches(20,30)
    fig.suptitle(output_label,fontsize = 16)
    fig.tight_layout(pad = 2)
    axs[4, 2].axis('off')
    row = 0
    col = 0
    species = outputs_df.columns[0:14]
    his_pars = []
    for s in species:
        
        y = outputs_df[s]
        
        ax = axs[row,col]
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)
            
        his_par = axs[row,col].hist(y,bins = 20)
        his_par = his_par[1]
        
        his_pars.append([s,his_par])
        axs[row,col].set_title(s+' (mg/L)',fontsize = 16)

        
        if col == 2:
            col = 0
            row += 1
        else:
            col +=1
    return(his_pars)

#%% run 1:1 plot function

bins = make_plots(abs_wq_df,'Hydroponics')
