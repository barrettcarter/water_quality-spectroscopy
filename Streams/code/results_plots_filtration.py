# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:01:15 2023

This script is for producing visualizations of the ML training and statistical
analysis results separate from the scripts which were used to produce those
results for the purpose of including these visualizations in my dissertation
and related publications.

Stream sample filtration experiment (Chapter 3)

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

path_to_wqs = r'D:\GitHub\PhD\water_quality-spectroscopy' # for external HD
# path_to_wqs = f'C:\\Users\\{user}\\Documents\\GitHub\\water_quality-spectroscopy' # for laptop

streams_output_dir = os.path.join(path_to_wqs,'Streams','outputs')

streams_fig_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\Stream results\\python'

#%% Make Streams boxplots for Chapter 3

abbrv = 'Streams-filtration'

test_rmses_path = os.path.join(streams_output_dir,'performance metrics', f'{abbrv}_test_rmse.csv')
test_rsqs_path = os.path.join(streams_output_dir,'performance metrics', f'{abbrv}_test_rsq.csv')

# make variables for plots (should only be run once)

perf_met_dict = {'test_rmse':pd.read_csv(test_rmses_path),
                 'test_rsq':pd.read_csv(test_rsqs_path)}

perf_mets = list(perf_met_dict.keys())

species = perf_met_dict['test_rmse'].species.unique()

fig_dir = streams_fig_dir

perf_met_labs = {'test_rmse':'log(test RMSE (mg/L))',
                 'test_rsq':'log(1 - test R-sq (unitless))'}

lev_trans = {'comb':'combined','fil':'filtered','unf':'unfiltered'}

labels_dict = {'Nitrate-N':'NO3-N','TKN':'TKN','ON':'ON','TN':'TN',
               'Ammonium-N':'NH4-N','Phosphate-P':'PO4-P','TP':'TP','OP':'OP'}

for pm in perf_mets:
    
    pm_df = perf_met_dict[pm]

    for lev in pm_df.treatment.unique():
        
        pm_df.loc[pm_df.treatment == lev,'treatment'] = lev_trans[lev]
        
    for sp in pm_df.species.unique():
        
        pm_df.loc[pm_df.species == sp,'species'] = labels_dict[sp]
        
    pm_df.rename(columns = {'trans_val':perf_met_labs[pm],
                            'model':'ML Algorithm',
                            'species':'Chemical Analyte'},inplace= True)

#%% bring in factor group letters

factor_letters_fn = f'{abbrv}_rsq-rmse_Tukey-by-spmod_factor-letters.csv'

factor_letters = pd.read_csv(os.path.join(streams_output_dir,'stats',factor_letters_fn))


#%% Make figure showing filtration results separately for each species, model, and perf_met

for pm in perf_mets:
    
    pm_df = perf_met_dict[pm]
    
    pm_lab = perf_met_labs[pm]
    
    g = sns.catplot(data = pm_df, y = pm_lab, x = 'Chemical Analyte', hue = 'treatment',
                    kind = 'box', palette = {'combined':'ghostwhite','filtered':'lightsteelblue',
                                             'unfiltered':'slategrey'},
                    row = 'ML Algorithm',height = 3,aspect = 2.5)
    
    plt.savefig(os.path.join(streams_fig_dir,f'filtration_{pm}_boxplot.png'),dpi = 300)

