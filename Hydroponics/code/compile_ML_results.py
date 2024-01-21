# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:15:51 2024

This script if for compiling all ML modeling results for each experiment to be 
used for statistical analyses and producing figures and tables.

@author: barre
"""

#%% import libraries

import pandas as pd

import numpy as np

import os
   
#%% set directories and bring in data

user = os.getlogin()

sample_type = 'Hydroponics' # used for navigating directories and other purposes

# proj_dir = 'D:/GitHub/PhD/water_quality-spectroscopy' # for work computer

proj_dir = r'C:\Users\barre\Documents\GitHub\water_quality-spectroscopy' # for laptop

output_dir = os.path.join(proj_dir, sample_type, 'outputs')

figure_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\HNS results\\python' # for work computer

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
  
#%% Save output

outputs_df.to_csv(os.path.join(output_dir,'HNSr_ML_results_compiled.csv'),index = False)
