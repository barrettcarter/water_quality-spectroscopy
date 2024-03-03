# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:13:09 2024

This script is for producing summary tables for HUM performance metrics separated
by chemical analyte, ML algorithm, and sample treatment

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

#%% Set paths

user = os.getlogin()

path_to_wqs = r'D:\GitHub\PhD\water_quality-spectroscopy' # for external HD
# path_to_wqs = f'C:\\Users\\{user}\\Documents\\GitHub\\water_quality-spectroscopy' # for laptop

perf_mets = ['test_rmse','test_rsq']

sample_types = ['Streams','Hydroponics']

exp_names = {'Streams':'Streams-filtration','Hydroponics':'HNS-dilution'}

fig_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images'

#%% bring in performance metric data for each sample type and save descriptive 
### stats for each performance metric, chemical analyte, and ML algorithm.

pm_summary = pd.DataFrame(columns = ['sample_type','species','model','treatment','perf_met',
                                     'mean','std','min','25%','50%','75%','max'])

pm = perf_mets[0]

stype = sample_types[0]

for pm in perf_mets:
    
    for stype in sample_types:
        
        exp_name = exp_names[stype]
        
        output_dir = os.path.join(path_to_wqs,stype,'outputs','performance metrics')
        
        output_fn = f'{exp_name}_{pm}.csv'
        
        pm_df = pd.read_csv(os.path.join(output_dir,output_fn))

        species = pm_df.species.unique()
        
        models = pm_df.model.unique()
        
        treatments = pm_df.treatment.unique()
        
        sp = species[0] # for testing
        
        mod = models[0] # for testing

        for sp in species:
            
            for mod in models:
                
                for trmt in treatments:
            
                    pm_smt = pm_df[(pm_df.species==sp)&(pm_df.model==mod)&(pm_df.treatment==trmt)]
                    
                    smt_stats = pd.DataFrame(pm_smt.value.describe()['mean':'max']).T.reset_index(drop=True)
                    
                    smt_stats[['sample_type','species','model','treatment','perf_met']] =\
                        [stype,sp,mod,trmt,pm]
                
                    pm_summary = pd.concat([pm_summary,smt_stats],ignore_index = True)
            
#%% save summary table

pm_summary.to_csv(os.path.join(path_to_wqs,'Outputs','perf_met_summary_sample_treatments.csv'),index = False)

#%% bring in performance metric data for each sample type and find the model with
### the minimum average RMSE and maximum average R-sq

pm_summary = pd.DataFrame(columns = ['sample_type','species','model','treatment','perf_met','value'])

pm = perf_mets[0]

stype = sample_types[0]

for pm in perf_mets:
    
    for stype in sample_types:
        
        output_dir = os.path.join(path_to_wqs,stype,'outputs','performance metrics')
        
        exp_name = exp_names[stype]
        
        output_fn = f'{exp_name}_{pm}.csv'
        
        pm_df = pd.read_csv(os.path.join(output_dir,output_fn))

        pm_df['smt'] = pm_df.species+'_'+pm_df.model+'_'+pm_df.treatment
        
        pm_df = pm_df.loc[:,['value','smt']]
        
        pm_means = pm_df.groupby('smt').mean()
        
        pm_means.reset_index(inplace = True)
        
        pm_means['species'] = pm_means.smt.apply(lambda x: x.split('_')[0])
        
        pm_means['model'] = pm_means.smt.apply(lambda x: x.split('_')[1])
        
        pm_means['treatment'] = pm_means.smt.apply(lambda x: x.split('_')[2])
        
        if pm == 'test_rmse':
        
            pm_opts = pm_means.loc[:,['value','species']].groupby('species').min().reset_index()
            
        elif pm == 'test_rsq':
            
            pm_opts = pm_means.loc[:,['value','species']].groupby('species').max().reset_index()
            
        for i,row in pm_opts.iterrows():
            
            mod_opt = pm_means.loc[(pm_means.species == row.species)&(pm_means.value==row.value),'model']
            
            trmt_opt = pm_means.loc[(pm_means.species == row.species)&(pm_means.value==row.value),'treatment']
            
            new_row = pd.DataFrame({'sample_type':stype,'species':row.species,'model':mod_opt,
                                    'treatment':trmt_opt,'perf_met':pm,'value':row.value})
            
            pm_summary = pd.concat([pm_summary,new_row],ignore_index = True)
            
#%% save summary table

pm_summary.to_csv(os.path.join(path_to_wqs,'Outputs','perf_met_opt_summary_sample_treatments.csv'),index = False)

#%% compile the true and predicted values for the optimal models

pm_summary = pd.read_csv(os.path.join(path_to_wqs,'Outputs','perf_met_opt_summary_sample_treatments.csv'))

pm_summary = pm_summary.loc[pm_summary.perf_met == 'test_rsq',:] # optimal models based on r-sq

# opt_ests = pd.DataFrame(columns = ['sample_type','species','model','y_true','y_pred'])

# proj_dir = path_to_wqs

stype = 'Streams' # for testing.
stype = sample_types[-1]

aliases = exp_names

for stype in sample_types:
    
    stype_al = aliases[stype]
    
    compiled_outputs = pd.read_csv(os.path.join(path_to_wqs,stype,'outputs',
                                                f'{stype_al}_ML_results_compiled.csv'))
    
    compiled_outputs.sample_type = stype
    
    stype_pm_opt = pm_summary.loc[pm_summary.sample_type==stype,:]
    
    valid_comb = stype_pm_opt.set_index(['sample_type','species','model','treatment']).index

    compiled_outputs = compiled_outputs[compiled_outputs.set_index(['sample_type','species','model',
                                                                    'treatment']).index.isin(valid_comb)]
    
    compiled_outputs = compiled_outputs[compiled_outputs.output.isin(['y_true_test','y_hat_test'])]
    
    if stype == sample_types[0]:
    
        true_tests = compiled_outputs
        
    else:
        
        true_tests = pd.concat([true_tests,compiled_outputs],ignore_index=True)
        
#%% reshape dataframe

st_map = {'Hydroponics':'HNS','Streams':'Stream'}

sp_map = {'Nitrate-N':'NO3-N','Potassium':'K','Calcium':'Ca','Sulfate':'SO4',
          'Phosphorus':'P','Magnesium':'Mg','Ammonium-N':'NH4-N',
          'pH':'pH','Iron':'Fe','Manganese':'Mn','Boron':'B',
          'Zinc':'Zn','Copper':'Cu','Molybdenum':'Mb',
          'TKN':'TKN','ON':'ON','TN':'TN',
          'Phosphate-P':'PO4-P','TP':'TP','OP':'OP'}

trmt_map = {'comb':'combined','fil':'filtered','unf':'unfiltered','diluted':'diluted',
            'undiluted':'undiluted'}

true_tests['sample_type'] = true_tests['sample_type'].apply(lambda x: st_map[x])

true_tests['species'] = true_tests['species'].apply(lambda x: sp_map[x])

true_tests['sample type, chemical analyte, ML algorithm, treatment'] = true_tests['sample_type']+\
    ', '+true_tests['species']+', '+true_tests['model']+', '+true_tests['treatment']

y_true_test = true_tests[true_tests.output=='y_true_test'].reset_index(drop=True)

y_hat_test = true_tests[true_tests.output=='y_hat_test'].reset_index(drop=True)

true_ests = y_true_test.copy()

true_ests.rename(columns = {'value':'True Concentration'},inplace=True)

true_ests['Estimated Concentration'] = y_hat_test['value']

ID_col = 'sample type, chemical analyte, ML algorithm, treatment'

true_ests.sort_values(by = ID_col,inplace=True)
#%% make 1:1 plots for best performing model/treatment for each chemical analyte/sample type

g = sns.lmplot(data = true_ests, x = 'True Concentration', y = 'Estimated Concentration',
           col = ID_col,col_wrap = 5,height = 3,
           facet_kws = {'sharey':False,'sharex':False},
           scatter_kws = {'color':'grey','alpha': 0.5,'s':2})

g.set_titles('{col_name}')

plt.savefig(os.path.join(fig_dir,'Exp2_opt_11_plots.png'),dpi = 300)

#%% make subset for 2 most significant/relevant effects of sample treatment

subs = ['Stream_TN','Stream_PO4-P','HNS_Fe','HNS_Zn']

true_ests_best = true_ests[(true_ests.sample_type+'_'+true_ests.species).isin(subs)]

true_ests_best.rename(columns = {'True Concentration':'True Concentration (mg/L)',
                                 'Estimated Concentration':'Estimated Concentration (mg/L)'},inplace = True)

#%% make 1:1 plots for best fits

plt.figure(dpi = 300)


g = sns.lmplot(data = true_ests_best, x = 'True Concentration (mg/L)', y = 'Estimated Concentration (mg/L)',
           col = ID_col,col_wrap = 2,height = 4,
           facet_kws = {'sharey':False,'sharex':False},
           scatter_kws = {'color':'grey','alpha': 0.5,'s':2})

g.set_titles('{col_name}')

plt.savefig(os.path.join(fig_dir,'Exp2_opt_best_11_plots.png'),dpi = 300)

#%% compile the true and predicted values for ALL models
### for making 1:1 plots showing all sample treatments

pm_summary = pd.read_csv(os.path.join(path_to_wqs,'Outputs','perf_met_opt_summary_sample_treatments.csv'))

pm_summary = pm_summary.loc[pm_summary.perf_met == 'test_rsq',:] # optimal models based on r-sq

# opt_ests = pd.DataFrame(columns = ['sample_type','species','model','y_true','y_pred'])

# proj_dir = path_to_wqs

stype = 'Streams' # for testing.
stype = sample_types[-1]

aliases = exp_names

for stype in sample_types:
    
    stype_al = aliases[stype]
    
    compiled_outputs = pd.read_csv(os.path.join(path_to_wqs,stype,'outputs',
                                                f'{stype_al}_ML_results_compiled.csv'))
    
    compiled_outputs.sample_type = stype
    
    stype_pm_opt = pm_summary.loc[pm_summary.sample_type==stype,:]
    
    valid_comb = stype_pm_opt.set_index(['sample_type','species','model']).index

    compiled_outputs = compiled_outputs[compiled_outputs.set_index(['sample_type','species','model']).index.isin(valid_comb)]
    
    compiled_outputs = compiled_outputs[compiled_outputs.output.isin(['y_true_test','y_hat_test'])]
    
    if stype == sample_types[0]:
    
        true_tests = compiled_outputs
        
    else:
        
        true_tests = pd.concat([true_tests,compiled_outputs],ignore_index=True)
        
#%% reshape dataframe

st_map = {'Hydroponics':'HNS','Streams':'Stream'}

sp_map = {'Nitrate-N':'NO3-N','Potassium':'K','Calcium':'Ca','Sulfate':'SO4',
          'Phosphorus':'P','Magnesium':'Mg','Ammonium-N':'NH4-N',
          'pH':'pH','Iron':'Fe','Manganese':'Mn','Boron':'B',
          'Zinc':'Zn','Copper':'Cu','Molybdenum':'Mb',
          'TKN':'TKN','ON':'ON','TN':'TN',
          'Phosphate-P':'PO4-P','TP':'TP','OP':'OP'}

trmt_map = {'comb':'combined','fil':'filtered','unf':'unfiltered','diluted':'diluted',
            'undiluted':'undiluted'}

true_tests['sample_type'] = true_tests['sample_type'].apply(lambda x: st_map[x])

true_tests['species'] = true_tests['species'].apply(lambda x: sp_map[x])

true_tests['treatment'] = true_tests['treatment'].apply(lambda x: trmt_map[x])

# true_tests['sample type, chemical analyte, ML algorithm, treatment'] = true_tests['sample_type']+\
#     ', '+true_tests['species']+', '+true_tests['model']+', '+true_tests['treatment']

true_tests['sample type, chemical analyte, model'] = true_tests['sample_type']+\
    ', '+true_tests['species']+', '+true_tests['model']

y_true_test = true_tests[true_tests.output=='y_true_test'].reset_index(drop=True)

y_hat_test = true_tests[true_tests.output=='y_hat_test'].reset_index(drop=True)

true_ests = y_true_test.copy()

true_ests.rename(columns = {'value':'True Concentration'},inplace=True)

true_ests['Estimated Concentration'] = y_hat_test['value']

ID_col = 'sample type, chemical analyte, model'

true_ests.sort_values(by = ID_col,inplace=True)

true_ests.rename(columns = {'treatment':'Sample Treatment'},inplace=True)
#%% make 1:1 plots for best performing model/treatment for each chemical analyte/sample type

g = sns.lmplot(data = true_ests, x = 'True Concentration', y = 'Estimated Concentration',
           col = ID_col,col_wrap = 5,height = 3,scatter = False,hue = 'Sample Treatment',
           facet_kws = {'sharey':False,'sharex':False},
           scatter_kws = {'color':'grey','alpha': 0.5,'s':2})

g.set_titles('{col_name}')

plt.savefig(os.path.join(fig_dir,'Exp2_ALL_11_plots.png'),dpi = 300)

#%% make subset for 2 most significant/relevant effects of sample treatment

subs = ['Stream_TN','Stream_PO4-P','HNS_Fe','HNS_Zn']

true_ests_best = true_ests[(true_ests.sample_type+'_'+true_ests.species).isin(subs)]

true_ests_best.rename(columns = {'True Concentration':'True Concentration (mg/L)',
                                 'Estimated Concentration':'Estimated Concentration (mg/L)'},inplace = True)

#%% make 1:1 plots for best fits

plt.figure(dpi = 300)


g = sns.lmplot(data = true_ests_best, x = 'True Concentration (mg/L)', y = 'Estimated Concentration (mg/L)',
           col = ID_col,col_wrap = 2,height = 4,
           facet_kws = {'sharey':False,'sharex':False},
           scatter_kws = {'color':'grey','alpha': 0.5,'s':2})

g.set_titles('{col_name}')

plt.savefig(os.path.join(fig_dir,'Exp2_opt_best_11_plots.png'),dpi = 300)

#%% Scratch

A = pd.DataFrame({'x':['a','b','c'],'y':['d','e','f'],'z':[1,2,3]})

B = pd.DataFrame({'x':['b','a','c','d','a','b','c','d'],
                  'y':['d','e','f','n','d','e','f','n'],
                  'z':[4,5,6,7,4,5,6,7]})

valid_comb = A.set_index(['x','y']).index

# B.loc[B.loc[:,['x','y']].isin(A.loc[:,['x','y']].to_dict(orient='list')).all(axis=1),:]

B[B.set_index(['x','y']).index.isin(valid_comb)]

###

valid_comb = stype_pm_opt.set_index(['sample_type','species','model']).index

compiled_outputs[compiled_outputs.set_index(['sample_type','species','model']).index.isin(valid_comb)]

# compiled_outputs.\
#     loc[:,['sample_type','species','model']].\
#         isin(stype_pm_opt.loc[:,['sample_type','species','model']].to_dict(orient='list')).\
#             all(axis=1)
            
# compiled_outputs.loc[(compiled_outputs.sample_type=='Streams')&\
#                      (compiled_outputs.species=='Ammonium-N')&\
#                          (compiled_outputs.model=='RF-PCA'),:].shape

# pd.Series(compiled_outputs.model.unique()).isin(stype_pm_opt.model.unique())
