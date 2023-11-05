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
import string

import matplotlib as mpl
plt.style.use('seaborn-whitegrid')

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

# proj_dir = 'D:/GitHub/PhD/water_quality-spectroscopy' # for work computer

proj_dir = r'C:\Users\barre\Documents\GitHub\water_quality-spectroscopy' # for work computer

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

abbrv = 'HNSr'

outputs = outputs_df.output.unique()

species = outputs_df.species.unique()

models = outputs_df.model.unique()

test_rmses = outputs_df.loc[outputs_df.output=='test_rmse',:]

test_rsqs = outputs_df.loc[outputs_df.output=='test_rsq',:]

#%% transform values to make more normally distributed

# take the log of rmse since it's natural range is [0,inf]

test_rmses['trans_val'] = np.log(test_rmses.value)

# transform rsq to log(1-rsq) since its natural range is [-inf,1]

test_rsqs['trans_val'] = np.log(1-test_rsqs.value)

#%% Do 2-way anova for both performance metrics.

rmse_anova_fmla = smf.ols('trans_val ~ C(model) + C(species) + C(model):C(species)',
                          data = test_rmses).fit()

sm.qqplot(rmse_anova_fmla.resid)

rsq_anova_fmla = smf.ols('trans_val ~ C(model) + C(species) + C(model):C(species)',
                          data = test_rsqs).fit()

sm.qqplot(rsq_anova_fmla.resid)

rmse_anova_rslt = sm.stats.anova_lm(rmse_anova_fmla,typ=2)

rsq_anova_rslt = sm.stats.anova_lm(rsq_anova_fmla,typ=2)

#%% save results

rmse_anova_rslt.to_csv(os.path.join(output_dir,'stats',f'{abbrv}_rmse_sp-mod-spmod_ANOVA.csv'))

rsq_anova_rslt.to_csv(os.path.join(output_dir,'stats',f'{abbrv}_rsq_sp-mod-spmod_ANOVA.csv'))

#%% perform pair-wise tukey test separated by species.

s = species[0] # for testing

perf_mets = {'test_rmse':test_rmses,'test_rsq':test_rsqs}

perf_met = 'test_rmse' # for testing

tukey_results = pd.DataFrame(columns = ['perf_met','species','mod1','mod2','meandiff',
                                    'pvalue','reject'])

mod_grps = pd.DataFrame(columns = ['perf_met','species','model','group_letters'])

grp_mods = pd.DataFrame(columns = ['perf_met','species','group_letter','models'])

for perf_met in list(perf_mets.keys()):
    
    pm_df = perf_mets[perf_met]

    for s in species:
        
        pm_sp = pm_df.loc[pm_df.species==s,:]
        
        pm_tukey = sm.stats.multicomp.pairwise_tukeyhsd(pm_sp.trans_val,
                                                        pm_sp.model)
        
        
        fig,ax = plt.subplots(figsize = [3.25,3.25])
        pm_tukey.plot_simultaneous(ax = ax, xlabel = f'trans({perf_met})', ylabel = 'model')
        ax.set_title(s)
        
        pm_mods = pm_tukey.groupsunique
        pm_pairinds = pm_tukey._multicomp.pairindices

        pm_mod1 = pm_mods[pm_pairinds[0]]
        pm_mod2 = pm_mods[pm_pairinds[1]]
        
        pm_results = pd.DataFrame(columns = tukey_results.columns)
        
        pm_results['mod1'] = pm_mod1
        pm_results['mod2'] = pm_mod2
        pm_results['meandiff'] = pm_tukey.meandiffs
        pm_results['pvalue'] = pm_tukey.pvalues
        pm_results['reject'] = pm_tukey.reject
        
        pm_results['perf_met'] = perf_met
        pm_results['species'] = s
        
        tukey_results = pd.concat([tukey_results,pm_results],ignore_index=True)
        
#%% Save results

tukey_results.to_csv(os.path.join(output_dir,'stats',
                                  f'{abbrv}_rsq-rmse_Tukey-by-sp.csv'),
                     index=False)