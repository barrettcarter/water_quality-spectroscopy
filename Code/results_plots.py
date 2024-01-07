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

# path_to_wqs = r'D:\GitHub\PhD\water_quality-spectroscopy' # for external HD
path_to_wqs = f'C:\\Users\\{user}\\Documents\\GitHub\\water_quality-spectroscopy' # for external HD


streams_output_dir = os.path.join(path_to_wqs,'Streams','outputs')

hns_output_dir = os.path.join(path_to_wqs,'Hydroponics','outputs')

hns_fig_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\HNS results\\python'

streams_fig_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\Stream results\\python'

#%% define plotting functions

def stats_group_plot(perf_met_dict = None, factor_col = None, block_col = None, 
                     factor_letters = None, factor_label = 'Factors', labels_dict = None,
                     save_fig = False, fig_dir = os.getcwd(), abbrv = None,
                     fig_dim = None):
    
    if perf_met_dict == None:
        
        raise ValueError('perf_met_dict must be defined as dictionary with keys being '+
              'the variable names of the performance metrics and values being '+
              'pandas dictionaries containing performance metrics values column '+
              'and factor and block columns.')
        
    if factor_col == None:
        
        raise ValueError('factor_col must be defined as name of column in perf_met_dict '+
              'containing factor levels correponding to each performance metric value.')
        
    if block_col == None:
        
        raise ValueError('block_col must be defined as name of column in perf_met_dict '+
              'containing block levels correponding to each performance metric value.')
        
    if type(factor_letters) != pd.core.frame.DataFrame:
        
        raise ValueError('factor_letters must be defined as a pandas dataframe '+
              'containing letters denoting post-hoc statistical significance groups. '+
              'This dataframe is generated by the post_hoc_groups function in the '+
              'output_stats.py script')
    
    if factor_label == 'Factors':
        
        print('factor_label can be specified to be used as xlabel in plot.')
        
    if labels_dict == None:
        
        raise ValueError('labels_dict must be defined as dictionary with keys identical to '+
              'performance metric variable names and values equal to the corresponding labels to '+
              'go on the plot.')
    
    if save_fig == True:
        
        if abbrv == None:
            
            print('abbrv could be specified to be used in saved figure file name.')
            
        if fig_dir == os.getcwd():
            
            print('figure will be saved to current working directory. specify other '+
                  'directory using fig_dir.')
            
    else:
            
        print('save figure using save_fig = True.')
        
    if fig_dim == None:
        
        raise ValueError('fig_dim must be specified as a string (square) or a list containing '+
              'number of rows and number of columns, in that order. ')
    
    
        
    perf_mets = list(perf_met_dict.keys())
    
    perf_met_labs = labels_dict
    
    for perf_met in perf_mets:
    
        pm_df = perf_met_dict[perf_met]
        
        block_levels = pm_df[block_col].unique()
        block_levels = np.sort(block_levels)
        
        factor_levels = pm_df[factor_col].unique()
        
        num_figs = len(block_levels)
    
        if fig_dim == 'square':
    
            fig_dim = int(num_figs**0.5)+1
            fig_dim = [fig_dim,fig_dim]
        
        perf_met_lab = perf_met_labs[perf_met]
    
        fig, axs = plt.subplots(nrows = fig_dim[0], ncols = fig_dim[1], figsize = (16,12),
                                dpi = 300, sharex = False)
        
        axs = fig.axes
        
        plt.subplots_adjust(hspace = 0.2)
        
        fig.text(0.5, 0.08, factor_label, ha='center', va='center',size=16) # figure x label
        fig.text(0.08, 0.5, perf_met_lab, ha='center', va='center', 
                 rotation='vertical',size=16) # figure y label
        
        row = 0 # for testing
        col = 0 # for testing
        
        i_block = 0 # required
        fig_num = 1 # required
        
        for row in range(fig_dim[0]):
            
            for col in range(fig_dim[1]):
                
                # ax = axs[row,col]
                ax = axs[fig_num-1]
                
                if fig_num > num_figs:
                    
                    ax.axis('off') # turn off extra axes
                    
                    if (row == fig_dim[0]-1) and (col == fig_dim[1]-1) and save_fig:
                    
                        fig_fn = f'{abbrv}_{perf_met}_F-{factor_col}_B-{block_col}_boxplot.png'
                            
                        plt.savefig(os.path.join(fig_dir,fig_fn))
                    
                    fig_num+=1
                    
                    continue
                
                block = block_levels[i_block]
            
                pm_block = pm_df.loc[pm_df[block_col]==block,:]
                
                pm_block = pm_block.sort_values(by = factor_col)
                
                letters_block_pm = factor_letters.loc[(factor_letters[block_col]==block)&
                                                      (factor_letters.perf_met==perf_met),
                                                      :].reset_index()
                
                y_text = pm_block['trans_val'].max() -\
                    0.1*(pm_block['trans_val'].max() -\
                         pm_block['trans_val'].min())
                        
                x_text = range(len(factor_levels))
                
                text = letters_block_pm.letters.to_numpy()
                
                # sns.violinplot(pm_block,x = 'model', y = 'trans_val',ax = ax, cut = 0)
                
                sns.boxplot(data = pm_block,x = factor_col, y = 'trans_val',ax = ax,
                            palette = 'Spectral_r')
                
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title(block)
                
                if ((row+1)*fig_dim[1] + (col+1)) <= num_figs:
                    
                    ax.set_xticklabels('')
                
                for i_text in range(len(text)):
                
                    ax.text(x_text[i_text],y_text,text[i_text],color = 'red',ha = 'center',
                            bbox = {'facecolor':'white','alpha':0.7,'boxstyle':'Round, pad=0.1'})
                
                if (row == fig_dim[0]-1) and (col == fig_dim[1]-1) and save_fig:
                    
                    fig_fn = f'{abbrv}_{perf_met}_F-{factor_col}_B-{block_col}_boxplot.png'
                        
                    plt.savefig(os.path.join(fig_dir,fig_fn))
                    
                    continue
                
                i_block +=1
                
                fig_num +=1
                
                if col == fig_dim[1] - 1:
                    
                    col = 0
                    row +=1
                
                else:
                    
                    col += 1

#%% Make HNS boxplots for Chapter 2

test_rmses_path = os.path.join(hns_output_dir,'performance metrics', 'test_rmse.csv')
test_rsqs_path = os.path.join(hns_output_dir,'performance metrics', 'test_rsq.csv')

# make variables for plots (should only be run once)

perf_met_dict = {'test_rmse':pd.read_csv(test_rmses_path),
                 'test_rsq':pd.read_csv(test_rsqs_path)}

perf_mets = list(perf_met_dict.keys())

species = perf_met_dict['test_rmse'].species.unique()

fig_dir = hns_fig_dir

abbrv = 'HNSr'

perf_met_labs = {'test_rmse':'log(test RMSE (mg/L))',
                 'test_rsq':'log(1 - test R-sq (unitless))'}

#%% Make figures for model groups

factor_letters_fn = 'HNSr_rsq-rmse_Tukey-by-sp_factor-letters.csv'

factor_letters = pd.read_csv(os.path.join(hns_output_dir,'stats',factor_letters_fn))

# they aren't saved here because they were saved when initially developing the plotting
# code in the output_stats.py script.

stats_group_plot(perf_met_dict = perf_met_dict, factor_col = 'model',block_col = 'species',
                 factor_letters = factor_letters, factor_label = 'ML Algorithm',
                 labels_dict = perf_met_labs,fig_dim = 'square',save_fig=True,
                 abbrv = abbrv, fig_dir = fig_dir)

#%% Make figures for species groups

factor_letters_fn = 'HNSr_rsq-rmse_Tukey-by-mod_factor-letters.csv'

factor_letters = pd.read_csv(os.path.join(hns_output_dir,'stats',factor_letters_fn))

# they aren't saved here because they were saved when initially developing the plotting
# code in the output_stats.py script.

stats_group_plot(perf_met_dict = perf_met_dict, factor_col = 'species',block_col = 'model',
                 factor_letters = factor_letters, factor_label = 'Chemical Analyte',
                 labels_dict = perf_met_labs,fig_dim = [4,1],save_fig=True,
                 abbrv = abbrv, fig_dir = fig_dir)