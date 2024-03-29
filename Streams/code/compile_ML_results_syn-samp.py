# -*- coding: utf-8 -*-
"""
Created on Sun Feb 08 2024

This script if for compiling all ML modeling results for the stream synthetic
sample experiment (Ch. 4). Results are used by other scripts
for statistical analyses and producing figures and tables.

@author: J. Barrett Carter
"""

#%% import libraries

import pandas as pd

import numpy as np

import os
   
#%% set directories and generate list of output files

user = os.getlogin()

sample_type = 'Streams' # used for navigating directories and other purposes

proj_dir = 'D:/GitHub/PhD/water_quality-spectroscopy' # for work computer

# proj_dir = r'C:\Users\barre\Documents\GitHub\water_quality-spectroscopy' # for laptop

output_dir = os.path.join(proj_dir, sample_type, 'outputs')

figure_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\Stream results\\python' # for work computer

# output_files = np.array(os.listdir(output_dir)) # not needed after saving file list

# output_files = pd.Series(output_files)

#%% choose which files to compile and save file list

# output_files = output_files[[19,21,22,24,25,27,28,30]] # select ML results files corresponding to experiment

# output_files.to_csv(os.path.join(output_dir,'stats','syn-samp_file_list.csv'))

#%% bring in data and concatenate

output_files = pd.read_csv(os.path.join(output_dir,'stats','syn-samp_file_list.csv'),
                           index_col = 0)['0']

output_files.reset_index(inplace=True,drop=True)

file = output_files[0]

outputs_df = pd.read_csv(os.path.join(output_dir,file))

stype_treatment = file.split('_')[0]

stype = stype_treatment.split('-')[0]

treatment = stype_treatment.split('-')[1]

syn_aug = (file.split('_')[1].split('-')[2])=='True'
 
outputs_df['sample_type'] = stype

# outputs_df['treatment'] = treatment # since all experiments involved filtered samples, this is not necessary

outputs_df['model'] = file.split('_')[2]

outputs_df['syn_aug'] = syn_aug

file = output_files[3]

for file in output_files[1:]:
  
  output_df = pd.read_csv(os.path.join(output_dir,file))
  
  output_df = output_df.loc[output_df.output.notna(),:] # get rid of empty rows

  stype_treatment = file.split('_')[0]

  stype = stype_treatment.split('-')[0]

  treatment = stype_treatment.split('-')[1]
  
  syn_aug = (file.split('_')[1].split('-')[2])=='True'
   
  output_df['sample_type'] = stype

  output_df['syn_aug'] = syn_aug

  output_df['model'] = file.split('_')[2]
  
  outputs_df = pd.concat([outputs_df,output_df],ignore_index = True)
  
#%% check dataframe

for col in outputs_df.columns:
    
    print(outputs_df[col].unique())
  
#%% Save output

outputs_df.to_csv(os.path.join(output_dir,'streams-syn-aug_ML_results_compiled.csv'),index = False)
