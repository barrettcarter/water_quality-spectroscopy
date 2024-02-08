# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:15:51 2024

This script if for compiling all ML modeling results for the general experiment
to be HUM performance experiment (Ch. 2). Results are used by other scripts
for statistical analyses and producing figures and tables.

@author: barre
"""

#%% import libraries

import pandas as pd

import numpy as np

import os
   
#%% set directories and bring in data

user = os.getlogin()

sample_type = 'Streams' # used for navigating directories and other purposes

# proj_dir = 'D:/GitHub/PhD/water_quality-spectroscopy' # for work computer

proj_dir = r'C:\Users\barre\Documents\GitHub\water_quality-spectroscopy' # for laptop

output_dir = os.path.join(proj_dir, sample_type, 'outputs')

figure_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\Stream results\\python' # for work computer

output_files = np.array(os.listdir(output_dir))

output_files = output_files[[44,48,49,53,63]] # select ML results files corresponding to experiment

file = output_files[0]

outputs_df = pd.read_csv(os.path.join(output_dir,file))

outputs_df['sample_type'] = file.split('_')[0]

outputs_df['model'] = file.split('_')[1]

for file in output_files[1:]:
  
  output_df = pd.read_csv(os.path.join(output_dir,file))
  
  output_df = output_df.loc[output_df.output.notna(),:] # get rid of empty rows

  output_df['sample_type'] = file.split('_')[0]

  output_df['model'] = file.split('_')[1]
  
  outputs_df = pd.concat([outputs_df,output_df],ignore_index = True)
  
#%% Save output

outputs_df.to_csv(os.path.join(output_dir,'streams_ML_results_compiled.csv'),index = False)
