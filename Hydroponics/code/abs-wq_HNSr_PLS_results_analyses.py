# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 08:58:22 2022

This script is for analyzing just the PLS results for the real hydroponic
nutrient solutions (HNSr) data.

@author: jbarrett.carter
"""

import pandas as pd
import os
#import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

#%% Set directory variables

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
output_dir = os.path.join(path_to_wqs,'Hydroponics/outputs/')
results_pls_fn = 'HNSr_PLS_It0-9_results_ALL-DATA.csv'
results_path = os.path.join(path_to_wqs,output_dir)

# sns.set_style("ticks")
# sns.set_palette('colorblind')
# sns.set(style = 'ticks',font_scale=2, palette = 'colorblind')

#%% Bring in data

results_pls_df=pd.read_csv(os.path.join(results_path,results_pls_fn))

#%% 1:1 plots function

def make_plots(outputs_df, output_label):
        
    fig, axs = plt.subplots(5,3)
    fig.set_size_inches(20,30)
    fig.suptitle(output_label,fontsize = 16)
    fig.tight_layout(pad = 2)
    axs[4, 2].axis('off')
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
        
        train_rsq = outputs_df['value'][(outputs_df.output == 'train_rsq')&
                            (outputs_df.species==s)]
        
        train_rsq = np.mean(train_rsq)
        
        test_rsq = outputs_df['value'][(outputs_df.output == 'test_rsq')&
                            (outputs_df.species==s)]
        
        test_rsq = np.mean(test_rsq)
        
        ax = axs[row,col]
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)
        
        axs[row,col].plot(y_true_train,y_hat_train,'o',markersize = 4, label = 'training set')
        axs[row,col].plot(y_true_test,y_hat_test,'o',markersize = 4, label = 'test set')
        axs[row,col].plot(line11,line11,'k--',label= '1:1 line')
        # axs[row,col].set_title(s)
        if (row == 0 and col == 0):
            axs[row,col].legend(loc = 'upper left',fontsize = 16)
        axs[row,col].set_xlabel('Lab Measured '+s+' (mg/L)',fontsize = 16)
        axs[row,col].set_ylabel('Predicted '+s+' (mg/L)',fontsize = 16)
        # axs[row,col].get_xaxis().set_visible(False)
        ax.text(x_text,y_text,r'$train\/r^2 =$'+str(np.round(train_rsq,3))+'\n'
                +r'$test\/r^2 =$'+str(np.round(test_rsq,3)), fontsize = 16)
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

#%% run 1:1 plot function

make_plots(results_pls_df,'Hydroponics')