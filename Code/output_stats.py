# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 08:36:17 2023

@author: carter_j
"""

#%% import libraries

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib as mpl
plt.style.use('seaborn-v0_8-whitegrid')

rc = {'axes.edgecolor':'0.1','axes.labelcolor':'0.1','grid.linestyle': '--',
      'text.color':'0.1','xtick.color':'0.1','ytick.color':'0.1','xtick.direction': 'in',
      'ytick.direction': 'in','patch.edgecolor': 'w','patch.force_edgecolor': True,
      'image.cmap': 'Set1','font.family': ['sans-serif'],
      'font.sans-serif': ['Arial','DejaVu Sans','Liberation Sans','Bitstream Vera Sans',
                          'sans-serif'],
      'xtick.bottom': True,'xtick.top': True,'ytick.left': True,'ytick.right': True,
      'axes.spines.left': True,'axes.spines.bottom': True,'axes.spines.right': True,
      'axes.spines.top': True,'figure.dpi': 300}

for rcparam in rc.keys():
    
    mpl.rcParams[rcparam] = rc[rcparam]
    
#%% set directories and bring in data

sample_type = 'Hydroponics' # used for navigating directories and other purposes

proj_dir = 'D:/GitHub/PhD/water_quality-spectroscopy' # for work computer

output_dir = os.path.join(proj_dir, sample_type, 'outputs')

figure_dir = 'C:\\Users\\carter_j\\OneDrive\\Research\\PhD\\Communications\\Images\\HNS results\\python' # for work computer

output_files = np.array(os.listdir(output_dir))

output_files = output_files[[13,16,19,20]] # select HNSr results

output_files = output_files[[1,2,3,0]] # re-order

file = output_files[0]

outputs_df = pd.read_csv(os.path.join(output_dir,file))

outputs_df['sample_type'] = file.split('_')[0]

outputs_df['model'] = file.split('_')[1]

for file in output_files[1:]:
  
  output_df = pd.read_csv(os.path.join(output_dir,file))

  output_df['sample_type'] = file.split('_')[0]

  output_df['model'] = file.split('_')[1]
  
  outputs_df = pd.concat([outputs_df,output_df],ignore_index = True)
  
#%% create some useful variables

outputs = outputs_df.output.unique()

species = outputs_df.species.unique()

test_rmses = outputs_df.loc[outputs_df.output=='test_rmse',:]

test_rsqs = outputs_df.loc[outputs_df.output=='test_rsq',:]

#%%