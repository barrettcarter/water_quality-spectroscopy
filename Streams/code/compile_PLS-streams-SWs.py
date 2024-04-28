# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:31:47 2024

This script was written for compiling the site-specific PLS modeling results
for comparing to synthetic surface water results (Chapter 4 of my dissertation).

@author: carter_j
"""

#%% Import libraries

import pandas as pd

import numpy as np
import os

import matplotlib.pyplot as plt

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

from matplotlib.cm import get_cmap

#%% Set paths

user = os.getlogin()

path_to_wqs = r'D:\GitHub\PhD\water_quality-spectroscopy' # for external HD
# path_to_wqs = f'C:\\Users\\{user}\\Documents\\GitHub\\water_quality-spectroscopy' # for laptop

perf_mets = ['test_rmse','test_rsq']

output_dir = os.path.join(path_to_wqs, 'Streams', 'outputs')

fig_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\Stream results\\python' # for work computer

# output_files = np.array(os.listdir(output_dir)) # not needed after saving file list

# output_files = pd.Series(output_files)

#%% choose which files to compile and save file list
### THIS DOESN'T NEED TO BE RUN AFTER THE FILE LIST HAS BEEN SAVED

# output_files = output_files[[32,33,34,35,36,37,38,40]] # select ML results files corresponding to experiment

# output_files.to_csv(os.path.join(output_dir,'stats','PLS-sites_file_list.csv'))

#%% compile results

output_files = pd.read_csv(os.path.join(output_dir,'stats','PLS-sites_file_list.csv'),
                           index_col = 0)['0']

output_files.reset_index(inplace=True,drop=True)

file = output_files[0]

outputs_df = pd.read_csv(os.path.join(output_dir,file))

outputs_df['site_ID'] = file.split('_')[0].split('-')[1]

for file in output_files[1:]:
  
  output_df = pd.read_csv(os.path.join(output_dir,file))
  
  output_df = output_df.loc[output_df.output.notna(),:] # get rid of empty rows

  output_df['site_ID'] = file.split('_')[0].split('-')[1]
  
  outputs_df = pd.concat([outputs_df,output_df],ignore_index = True)
  
#%% Save output

outputs_df.to_csv(os.path.join(output_dir,'PLS-streams-SWs_results_compiled.csv'),index = False)
