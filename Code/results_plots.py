# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:01:15 2023

This script is for producing visualizations of the ML training and statistical
analysis results separate from the scripts which were used to produce those
results for the purpose of including these visualizations in my dissertation
and related publications.

@author: J. Barrett Carter
"""
#%% Import libraries

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import os

import seaborn as sns

sns.set_theme(style = 'whitegrid',font_scale=1)
#sns.set(font_scale=2)

import matplotlib as mpl
# plt.style.use('seaborn-whitegrid')

rc = {'axes.edgecolor':'0.1','axes.labelcolor':'0.1','grid.linestyle': '--',
      'text.color':'0.1','xtick.color':'0.1','ytick.color':'0.1','xtick.direction': 'in',
      'ytick.direction': 'in','patch.edgecolor': 'w','patch.force_edgecolor': True,
      'image.cmap': 'Set1','font.family': ['sans-serif'],
      'font.sans-serif': ['Arial','DejaVu Sans','Liberation Sans','Bitstream Vera Sans',
                          'sans-serif'],
      'axes.spines.left': True,'axes.spines.bottom': True,'axes.spines.right': True,
      'axes.spines.top': True,'figure.dpi': 300}

for rcparam in rc.keys():
    
    mpl.rcParams[rcparam] = rc[rcparam]

sns.set_style(rc = rc)

#%% Set paths

user = os.getlogin()

path_to_wqs = r'D:\GitHub\PhD\water_quality-spectroscopy'

streams_output_dir = os.path.join(path_to_wqs,'Streams','outputs')

hns_output_dir = os.path.join(path_to_wqs,'Hydroponics','outputs')

hns_fig_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\HNS results\\python'

streams_fig_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\Stream results\\python'

#%% Make HNS boxplots for Chapter 2



# make variables for plots (should only be run once)

perf_met_dict = perf_mets

perf_mets = list(perf_met_dict.keys())

#%% Make figures

perf_met_labs = {'test_rmse':'log(test RMSE (mg/L))',
                 'test_rsq':'log(1 - test R-sq (unitless))'}

num_figs = len(species)

fig_dim = int(num_figs**0.5)+1

for perf_met in perf_mets:
    
    pm_df = perf_met_dict[perf_met]
    
    perf_met_lab = perf_met_labs[perf_met]

    fig, axs = plt.subplots(nrows = fig_dim, ncols = fig_dim, figsize = (16,12),
                            dpi = 300, sharex = False)
    
    plt.subplots_adjust(hspace = 0.2)
    
    fig.text(0.5, 0.08, 'ML Algorithm', ha='center', va='center',size=16) # figure x label
    fig.text(0.08, 0.5, perf_met_lab, ha='center', va='center', 
             rotation='vertical',size=16) # figure y label
    
    row = 0 # for testing
    col = 0 # for testing
    
    i_sp = 0 # required
    fig_num = 1 # required
    
    for row in range(fig_dim):
        
        for col in range(fig_dim):
            
            ax = axs[row,col]
            
            if fig_num > num_figs:
                
                ax.axis('off') # turn off extra axes
                
                if fig_num == fig_dim**2:
                    
                    plt.savefig(os.path.join(figure_dir,f'{abbrv}_{perf_met}_boxplot.png'))
                
                fig_num+=1
                
                continue
            
            sp = species[i_sp]
        
            pm_sp = pm_df.loc[pm_df.species==sp,:]
            
            pm_sp = pm_sp.sort_values(by = 'model')
            
            letters_sp_pm = factor_letters.loc[(factor_letters.species==sp)&
                                               (factor_letters.perf_met==perf_met),:].reset_index()
            
            y_text = pm_sp['trans_val'].max() -\
                0.1*(pm_sp['trans_val'].max() -\
                     pm_sp['trans_val'].min())
                    
            x_text = [0,1,2,3]
            
            text = letters_sp_pm.letters.to_numpy()
            
            # sns.violinplot(pm_sp,x = 'model', y = 'trans_val',ax = ax, cut = 0)
            
            sns.boxplot(data = pm_sp,x = 'model', y = 'trans_val',ax = ax)
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(sp)
            
            if ((row+1)*4 + (col+1)) <= num_figs:
                
                ax.set_xticklabels('')
            
            for i_text in range(len(text)):
            
                ax.text(x_text[i_text],y_text,text[i_text],color = 'red',ha = 'center',
                        bbox = {'facecolor':'white','alpha':0.7,'boxstyle':'Round, pad=0.1'})
            
            i_sp +=1
            
            fig_num +=1
            
            if col == fig_dim - 1:
                
                col = 0
                row +=1
            
            else:
                
                col += 1