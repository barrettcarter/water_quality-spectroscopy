# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:01:15 2023

This script is for producing visualizations of the ML training and statistical
analysis results separate from the scripts which were used to produce those
results for the purpose of including these visualizations in my dissertation
and related publications.

HNS synthetic sample augmentation experiment (Chapter 4)

@author: J. Barrett Carter
"""
#%% Import libraries

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import os

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

output_dir = os.path.join(path_to_wqs,'Hydroponics','outputs')

fig_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images\\HNS results\\python'

#%% Pring in performance metric data and create useful variables

treatment = 'diluted'

abbrv = f'HNS-{treatment}-syn-aug'

test_rmses_path = os.path.join(output_dir,'performance metrics', f'{abbrv}_test_rmse.csv')
test_rsqs_path = os.path.join(output_dir,'performance metrics', f'{abbrv}_test_rsq.csv')

# make variables for plots (should only be run once)

perf_met_dict = {'test_rmse':pd.read_csv(test_rmses_path),
                 'test_rsq':pd.read_csv(test_rsqs_path)}

perf_mets = list(perf_met_dict.keys())

species = perf_met_dict['test_rmse'].species.unique()

fig_dir = fig_dir

perf_met_labs = {'test_rmse':'log(test RMSE (mg/L))',
                 'test_rsq':'log(1 - test R-sq (unitless))'}

labels_dict = {'Nitrate-N':'NO3-N','Potassium':'K','Calcium':'Ca','Sulfate':'SO4',
               'Phosphorus':'P','Magnesium':'Mg','Ammonium-N':'NH4-N',
               'pH':'pH','Iron':'Fe','Manganese':'Mn','Boron':'B',
               'Zinc':'Zn','Copper':'Cu','Molybdenum':'Mb'}

spcol = 'Chemical Analyte'
modcol = 'ML Algorithm'
syncol = 'syn. aug.'

for pm in perf_mets:
    
    pm_df = perf_met_dict[pm]

    pm_df['syn_aug'] = pm_df.syn_aug.apply(lambda x: str(x))
        
    for sp in pm_df.species.unique():
        
        pm_df.loc[pm_df.species == sp,'species'] = labels_dict[sp]
        
    pm_df['species_model'] = pm_df.species +'_' + pm_df.model
        
    pm_df.rename(columns = {'trans_val':perf_met_labs[pm],
                            'model':modcol,
                            'species':spcol,
                            'syn_aug':syncol},inplace= True)

#%% bring in factor group letters

factor_letters_fn = f'{abbrv}_rsq-rmse_Tukey-by-spmod_factor-letters.csv'

factor_letters = pd.read_csv(os.path.join(output_dir,'stats',factor_letters_fn))

# modify species names to match labels in figure

factor_letters['species'] = factor_letters.species_model.apply(lambda x: x.split('_')[0])

factor_letters['model'] = factor_letters.species_model.apply(lambda x: x.split('_')[1])

for sp in list(labels_dict.keys()):
    
    factor_letters.loc[factor_letters.species==sp,'species'] = labels_dict[sp]

#%% Make figure showing syn-aug results separately for each species, model, and perf_met

pm = perf_mets[0] # for testing

for pm in perf_mets:
    
    pm_df = perf_met_dict[pm]
    
    pm_lab = perf_met_labs[pm]
    
    pm_max = pm_df[pm_lab].max()
    
    ### make figure using seaborn
    
    g = sns.catplot(data = pm_df, y = pm_lab, x = spcol, hue = syncol,
                    kind = 'box', palette = {'False':'ghostwhite','True':'lightsteelblue'},
                    row = modcol,height = 3,aspect = 2.5)
    
    ### loop through each axis and block-factor level to add Tukey significance letters
    
    for imod,ax in enumerate(g.figure.axes):
        
        mod = pm_df[modcol].unique()[imod]
        
        for isp,sp in enumerate(pm_df[spcol].unique()):
            
            y_text = pm_max
    
            x_text = [isp-0.25,isp,isp+0.25]
        
            text = factor_letters.loc[(factor_letters.model==mod)&\
                                      (factor_letters.species==sp)&\
                                          (factor_letters.perf_met==pm),'letters'].values
        
            for i_text in range(len(text)):
            
                ax.text(x_text[i_text],y_text,text[i_text],color = 'red',ha = 'center',
                        bbox = {'facecolor':'white','alpha':0.7,'boxstyle':'Round, pad=0.1'})
    
    ### save figure
    
    plt.savefig(os.path.join(fig_dir,f'{abbrv}_{pm}_boxplot.png'),dpi = 300)

