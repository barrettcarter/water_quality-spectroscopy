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

import seaborn as sns

sns.set_theme(style = 'whitegrid',font_scale=1)
#sns.set(font_scale=2)
sns.set_style(rc = {'axes.edgecolor':'0.1',
                      'axes.labelcolor':'0.1',
                      'grid.linestyle': '--',
                      'text.color':'0.1',
                      'xtick.color':'0.1',
                      'ytick.color':'0.1',
                      'xtick.direction': 'in',
                      'ytick.direction': 'in',
                      'patch.edgecolor': 'w',
                      'patch.force_edgecolor': True,
                      'image.cmap': 'Set1',
                      'font.family': ['sans-serif'],
                      'font.sans-serif': ['Arial',
                      'DejaVu Sans',
                      'Liberation Sans',
                      'Bitstream Vera Sans',
                      'sans-serif'],
                      'xtick.bottom': True,
                      'xtick.top': True,
                      'ytick.left': True,
                      'ytick.right': True,
                      'axes.spines.left': True,
                      'axes.spines.bottom': True,
                      'axes.spines.right': True,
                      'axes.spines.top': True})

import matplotlib as mpl
# plt.style.use('seaborn-whitegrid')

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

# proj_dir = r'C:\Users\barre\Documents\GitHub\water_quality-spectroscopy' # for laptop

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

alphabet = string.ascii_lowercase

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
        
#%% Define grouping function

# define variables for testing

results_df = tukey_results
factor_cols = ['mod1','mod2']
block_col = 'species'
conclusion_col = 'reject'
response_col = 'perf_met'

def post_hoc_groups(results_df,factor_cols,block_col,conclusion_col,
                    response_col):
    
    results_df = results_df.copy()
    
    # make list of unique block labels
    
    blocks = results_df[block_col].unique()
        
    # make list of unique factor labels
        
    factor_levs = results_df[factor_cols[0]]
    
    for factor_col in factor_cols[1:]:
    
        factor_levs = pd.concat([factor_levs,results_df[factor_col]],
                           ignore_index=True)
        
    factor_levs = factor_levs.unique()
    
    responses = results_df[response_col].unique()
    
    ID_cols = pd.concat([pd.Series(block_col),pd.Series(response_col)])
    
    fl_cols = pd.concat([ID_cols,pd.Series(['factor_lev','letters'])])
    
    lf_cols = pd.concat([ID_cols,pd.Series(['letter','factor_levs'])])
    
    factor_letters = pd.DataFrame(columns = fl_cols)
    
    for response in responses:
        
        for block in blocks:
            
            for factor_lev in factor_levs:
                
                new_row = pd.DataFrame(columns = fl_cols)
                
                new_row.loc[0,block_col:'factor_lev'] = [block, response, factor_lev]
                
                factor_letters = pd.concat([factor_letters,new_row],ignore_index=True)
    
    factor_letters['letters']='A'
    
    letter_factors = pd.DataFrame(columns = lf_cols)
    
    response = responses[0] # for testing

    for response in responses:
        
        res_sub = results_df.loc[results_df[response_col]==response,:].reset_index(drop = True)
        
        res_sig = res_sub.loc[res_sub[conclusion_col],:]
        
        res_ins = res_sub.loc[res_sub[conclusion_col]==False,:]
        
        block = blocks[0] # for testing
        
        block = 'Potassium' # for testing
        
        for block in blocks:
            
            li = 0   # reset grouping letters
            
            res_ins_bl = res_ins.loc[res_ins[block_col]==block,:]
            
            res_sig_bl = res_sig.loc[res_sig[block_col]==block,:]
            
            factor_lev = factor_levs[0] # for testing
            
            # factor_lev = 'PLS' # for testing
            
            for factor_lev in factor_levs:
                
                letter = alphabet[li]
                
                # see if factor level was in any non-significantly different pairs
                
                if (res_ins_bl[factor_cols]==factor_lev).any(axis = None):
                    
                    res_ins_bl_f = res_ins_bl.loc[(res_ins_bl[factor_cols]==factor_lev).any(axis = 1),:]
            
                    group_fac_levs = np.unique(res_ins_bl_f[factor_cols].values).flatten()
                    
                    sig_pairs_cond = (res_sig_bl[factor_cols].isin(group_fac_levs)).all(axis=1)
                    
                    sig_pairs = res_sig_bl.loc[sig_pairs_cond,factor_cols]
                    
                    i = 0
                    
                    while sig_pairs.shape[0]>0:
                        
                        sig_fac_levs = np.unique(sig_pairs.values).flatten()
                    
                    fac_lev_drop = np.unique(sig_pairs.values).flatten()
                    
                    fac_lev_drop = fac_lev_drop[fac_lev_drop != factor_lev]
                    
                    group_fac_levs = group_fac_levs[np.isin(group_fac_levs,fac_lev_drop)==False]
                    
                    group_fac_levs = np.sort(group_fac_levs)
                    
                    # if only one is left, that means the group is redundant
                    # need to move on to next iteration
                    
                    if len(group_fac_levs)==1:
                        
                        continue
    
                        
                else: 
                    
                    # In the case where the factor level is different from all others, it's its own group
                    
                    group_fac_levs = np.array([factor_lev])
                    
                # # put group_factors into a series for putting into the letter_factors dataframe
                
                ### This may not be the best way to go about it
                    
                # gf_sr = pd.Series(np.array([]),dtype=object)
                        
                # gf_sr[0] = group_fac_levs
                
                
                # make letter group into a string
                
                gf_str = ' '.join(group_fac_levs)
                
                if letter_factors.shape[0]==0: # for the first iteration
                    
                    new_row_lf = pd.DataFrame(columns = letter_factors.columns)
                    
                    # new_row_lf.loc[0,:] = [block,response,letter,gf_sr[0]]
                    
                    new_row_lf.loc[0,:] = [block,response,letter,gf_str]
                    
                    letter_factors = pd.concat([letter_factors,new_row_lf],ignore_index=True)
                    
                    fac_let_rows = (factor_letters[block_col] == block)&\
                        (factor_letters[response_col] == response)&\
                            (factor_letters.factor_lev.isin(group_fac_levs))
                    
                    factor_letters.loc[fac_let_rows,'letters'] =\
                        factor_letters.loc[fac_let_rows,'letters']+letter
                        
                    li += 1
                
                elif (letter_factors.loc[(letter_factors[block_col] == block)&\
                        (letter_factors[response_col] == response),'factor_levs']==gf_str).sum()==0: # making sure group doesdn't already exists
                    
                    new_row_lf = pd.DataFrame(columns = letter_factors.columns)
                    
                    # new_row_lf.loc[0,:] = [block,response,letter,gf_sr[0]]
                    new_row_lf.loc[0,:] = [block,response,letter,gf_str]
                    
                    letter_factors = pd.concat([letter_factors,new_row_lf],ignore_index=True)
                    
                    fac_let_rows = (factor_letters[block_col] == block)&\
                        (factor_letters[response_col] == response)&\
                            (factor_letters.factor_lev.isin(group_fac_levs))
                    
                    factor_letters.loc[fac_let_rows,'letters'] =\
                        factor_letters.loc[fac_let_rows,'letters']+letter
                    
                    li += 1
                    
    factor_letters['letters'] = factor_letters.letters.apply(lambda x: x[1:])
    
    return({'letter_factors':letter_factors,'factor_letters':factor_letters})
                    
#%% run grouping function

grouping_dict = post_hoc_groups(results_df = tukey_results,factor_cols = ['mod1','mod2'],
                                block_col = 'species',
                                conclusion_col = 'reject',
                                response_col = 'perf_met')    

factor_letters = grouping_dict['factor_letters']   
letter_factors = grouping_dict['letter_factors']

#%% Save results

tukey_results.to_csv(os.path.join(output_dir,'stats',
                                  f'{abbrv}_rsq-rmse_Tukey-by-sp.csv'),
                     index=False)

factor_letters.to_csv(os.path.join(output_dir,'stats',
                                  f'{abbrv}_rsq-rmse_Tukey-by-sp_factor-letters.csv'),
                     index=False)

letter_factors.to_csv(os.path.join(output_dir,'stats',
                                  f'{abbrv}_rsq-rmse_Tukey-by-sp_letter-factors.csv'),
                     index=False)

#%% make variables for plots (should only be run once)

perf_met_dict = perf_mets

perf_mets = list(perf_met_dict.keys())

#%% Make figures

perf_met_labs = {'test_rmse':'log(test RMSE (mg/L))',
                 'test_rsq':'log(1 - test R-sq (unitless))'}

num_figs = len(species)

fig_dim = int(num_figs**0.5)+1

for perf_met in perf_mets:
    
    pm_df = perf_met_dict[perf_met]
    
    perf_met_lab = perf_met_labs[perf_met]

    fig, axs = plt.subplots(nrows = fig_dim, ncols = fig_dim, figsize = (16,12),
                            dpi = 300, sharex = False)
    
    plt.subplots_adjust(hspace = 0.2)
    
    fig.text(0.5, 0.08, 'ML Algorithm', ha='center', va='center',size=16) # figure x label
    fig.text(0.08, 0.5, perf_met_lab, ha='center', va='center', 
             rotation='vertical',size=16) # figure y label
    
    row = 0 # for testing
    col = 0 # for testing
    
    i_sp = 0 # required
    fig_num = 1 # required
    
    for row in range(fig_dim):
        
        for col in range(fig_dim):
            
            ax = axs[row,col]
            
            if fig_num > num_figs:
                
                ax.axis('off') # turn off extra axes
                
                if fig_num == fig_dim**2:
                    
                    plt.savefig(os.path.join(figure_dir,f'{abbrv}_{perf_met}_boxplot.png'))
                
                fig_num+=1
                
                continue
            
            sp = species[i_sp]
        
            pm_sp = pm_df.loc[pm_df.species==sp,:]
            
            pm_sp = pm_sp.sort_values(by = 'model')
            
            letters_sp_pm = factor_letters.loc[(factor_letters.species==sp)&
                                               (factor_letters.perf_met==perf_met),:].reset_index()
            
            y_text = pm_sp['trans_val'].max() -\
                0.1*(pm_sp['trans_val'].max() -\
                     pm_sp['trans_val'].min())
                    
            x_text = [0,1,2,3]
            
            text = letters_sp_pm.letters.to_numpy()
            
            # sns.violinplot(pm_sp,x = 'model', y = 'trans_val',ax = ax, cut = 0)
            
            sns.boxplot(pm_sp,x = 'model', y = 'trans_val',ax = ax)
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(sp)
            
            if ((row+1)*4 + (col+1)) <= num_figs:
                
                ax.set_xticklabels('')
            
            for i_text in range(len(text)):
            
                ax.text(x_text[i_text],y_text,text[i_text],color = 'red',ha = 'center',
                        bbox = {'facecolor':'white','alpha':0.7,'boxstyle':'Round, pad=0.1'})
            
            i_sp +=1
            
            fig_num +=1
            
            if col == fig_dim - 1:
                
                col = 0
                row +=1
            
            else:
                
                col += 1