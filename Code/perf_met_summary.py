# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:13:09 2024

This script is for producing summary tables for HUM performance metrics separated
by chemical analyte and ML algorithm

@author: carter_j
"""

#%% Import libraries

import pandas as pd

import numpy as np
import os

#%% Set paths

user = os.getlogin()

path_to_wqs = r'D:\GitHub\PhD\water_quality-spectroscopy' # for external HD
# path_to_wqs = f'C:\\Users\\{user}\\Documents\\GitHub\\water_quality-spectroscopy' # for laptop

perf_mets = ['test_rmse','test_rsq']

sample_types = ['Streams','Hydroponics']

#%% bring in performance metric data for each sample type and find the model with
### the minimum average RMSE and maximum average R-sq

pm_summary = pd.DataFrame(columns = ['sample_type','species','model','perf_met','value'])

pm = perf_mets[0]

stype = sample_types[0]

for pm in perf_mets:
    
    for stype in sample_types:
        
        output_dir = os.path.join(path_to_wqs,stype,'outputs','performance metrics')
        
        output_fn = f'{stype}_{pm}.csv'
        
        pm_df = pd.read_csv(os.path.join(output_dir,output_fn))

        pm_df['spmod'] = pm_df.species+'_'+pm_df.model
        
        pm_df = pm_df.loc[:,['value','spmod']]
        
        pm_means = pm_df.groupby('spmod').mean()
        
        pm_means.reset_index(inplace = True)
        
        pm_means['species'] = pm_means.spmod.apply(lambda x: x.split('_')[0])
        
        pm_means['model'] = pm_means.spmod.apply(lambda x: x.split('_')[1])
        
        if pm == 'test_rmse':
        
            pm_opts = pm_means.loc[:,['value','species']].groupby('species').min().reset_index()
            
        elif pm == 'test_rsq':
            
            pm_opts = pm_means.loc[:,['value','species']].groupby('species').max().reset_index()
            
        for i,row in pm_opts.iterrows():
            
            mod_opt = pm_means.loc[(pm_means.species == row.species)&(pm_means.value==row.value),'model']
            
            new_row = pd.DataFrame({'sample_type':stype,'species':row.species,'model':mod_opt,
                                    'perf_met':pm,'value':row.value})
            
            pm_summary = pd.concat([pm_summary,new_row],ignore_index = True)
            
#%% save summary table

pm_summary.to_csv(os.path.join(path_to_wqs,'Outputs','perf_met_opt_summary.csv'),index = False)

#%% compile the true and predicted values for the optimal models

opt_ests = pd.DataFrame(columns = ['sample_type','species','model','y_true','y_pred'])

proj_dir = path_to_wqs

stype = 'Streams' # for testing.

for stype in sample_types:
    
    sample_type = stype
    
    st_rsq = pm_summary.loc[(pm_summary.perf_met=='test_rsq')&(pm_summary.sample_type==stype),:]
    
    spmod_opts = st_rsq.species + '_' + st_rsq.model
    
    st_rsq['spmod'] = spmod_opts

    output_dir = os.path.join(proj_dir, sample_type, 'outputs')
    
    output_files = np.array(os.listdir(output_dir))
    
    output_files = output_files[[44,48,49,53,63]] # select ML results files corresponding to experiment
    
    file = output_files[0]
    
    outputs_df = pd.DataFrame(columns = pd.read_csv(os.path.join(output_dir,file)).columns)
    
    for file in output_files:
        
        file_mod = file.split('_')[1]
        
        if (st_rsq.model==file_mod).any() == False:
            
            continue
        
        output_df = pd.read_csv(os.path.join(output_dir,file))
        
        output_df = output_df.loc[output_df.output.notna(),:] # get rid of empty rows
          
        output_df['sample_type'] = file.split('_')[0]
          
        output_df['model'] = file.split('_')[1]
        
        outputs_df = pd.concat([outputs_df,output_df],ignore_index = True)
        
        output_df['spmod'] = output_df.species + '_' + output_df.model
        
        output_df = output_df.loc[output_df.spmod.isin(spmod_opts),:]
