# -*- coding: utf-8 -*-
"""
Created on Sun Feb 08 2024

This script if for compiling all ML modeling results for the HNS synthetic
sample experiment (Ch. 3). Results are used by other scripts
for statistical analyses and producing figures and tables.

@author: J. Barrett Carter
"""

#%% import libraries

import pandas as pd

import numpy as np

import os
   
#%% set directories and generate list of output files

user = os.getlogin()

sample_type = 'Hydroponics' # used for navigating directories and other purposes

proj_dir = 'D:/GitHub/PhD/water_quality-spectroscopy' # for work computer

# proj_dir = r'C:\Users\barre\Documents\GitHub\water_quality-spectroscopy' # for laptop

output_dir = os.path.join(proj_dir, sample_type, 'outputs')

figure_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\Stream results\\python' # for work computer

output_files = np.array(os.listdir(output_dir)) # not needed after saving file list

output_files = pd.Series(output_files)

#%% choose which files to compile and save file list

# output_files = output_files[[1,2,3,4,5,6,7,8,25,26,27,28,29,30,31,32]] # select ML results files corresponding to experiment

output_files = pd.concat([output_files[1:9],
                          output_files[10:26],
                          output_files[42:50]]) # select ML results files corresponding to experiment

output_files.to_csv(os.path.join(output_dir,'stats','syn-samp_file_list.csv'))

#%% bring in data and concatenate

output_files = pd.read_csv(os.path.join(output_dir,'stats','syn-samp_file_list.csv'),
                           index_col = 0)['0']

output_files.reset_index(inplace=True,drop=True)

file = output_files[0]

outputs_df = pd.read_csv(os.path.join(output_dir,file))

stype_treatment = file.split('_')[0]

stype = 'HNS'

if 'd30' in stype_treatment:
    
    treatment = 'diluted'
    
else:
    
    treatment = 'undiluted'
    
syn_aug = (file.split('_')[1].split('-')[2])=='True'
 
outputs_df['sample_type'] = stype

outputs_df['treatment'] = treatment

outputs_df['model'] = file.split('_')[2]

outputs_df['syn_aug'] = syn_aug

if 'SO4' not in file:
    
    outputs_df = outputs_df.loc[outputs_df.species != 'Sulfate',:]

for file in output_files[1:]:
    
    output_df = pd.read_csv(os.path.join(output_dir,file))
  
    output_df = output_df.loc[output_df.output.notna(),:] # get rid of empty rows
    
    stype_treatment = file.split('_')[0]

    stype = 'HNS'

    if 'd30' in stype_treatment:
        
        treatment = 'diluted'
        
    else:
        
        treatment = 'undiluted'
        
    syn_aug = (file.split('_')[1].split('-')[2])=='True'
        
    output_df['sample_type'] = stype
    
    output_df['treatment'] = treatment

    output_df['model'] = file.split('_')[2]
    
    output_df['syn_aug'] = syn_aug
    
    if 'SO4' not in file:
        
        output_df = output_df.loc[output_df.species != 'Sulfate',:]
        
    else:
        
        output_df = output_df.loc[output_df.species == 'Sulfate',:]
        
    if 'Nitrate-N' not in output_df.species.unique():
        
        print(f'Nitrate-N missing in {file}')
  
    outputs_df = pd.concat([outputs_df,output_df],ignore_index = True)
    
#%% check df

print(outputs_df.columns)
print(outputs_df.isna().sum())
for col in outputs_df.columns:
    
    print(outputs_df[col].unique())
    
    print(outputs_df.loc[outputs_df.species=='Nitrate-N',col].unique())
    
print(outputs_df.loc[outputs_df.species=='Sulfate',:])

print(outputs_df.species.value_counts())
  
#%% Save output

outputs_df.to_csv(os.path.join(output_dir,'HNS-syn-aug_ML_results_compiled.csv'),index = False)
