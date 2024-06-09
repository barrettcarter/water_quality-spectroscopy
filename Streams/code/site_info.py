# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:50:54 2024

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

#%% Set paths and bring in data

user = os.getlogin()

path_to_wqs = r'D:\GitHub\PhD\water_quality-spectroscopy' # for external HD
# path_to_wqs = f'C:\\Users\\{user}\\Documents\\GitHub\\water_quality-spectroscopy' # for laptop

site_info = pd.read_csv(os.path.join(path_to_wqs,'Streams','inputs','LULC_watersheds.csv'))

fig_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images'

#%% data wrangling and stats

lulc = site_info.loc[:,['LEVEL1_L_1','Name','Area_sq_ft','Shape_Area']].rename(columns = {'LEVEL1_L_1':'lulc'})

A_totals = lulc.loc[:,['Name','Area_sq_ft']].groupby('Name').sum()

SA_totals = lulc.loc[:,['Name','Shape_Area']].groupby('Name').sum()

A_totals['Shape_Area'] = SA_totals

A_totals['ratio'] = A_totals.Area_sq_ft/A_totals.Shape_Area

lulc_totals = lulc.loc[:,['Name','lulc','Shape_Area']].groupby(['Name','lulc']).sum()

lulc_totals.reset_index(inplace=True)

A_totals.reset_index(inplace=True)

lulc_totals = pd.merge(lulc_totals,A_totals,on='Name',how = 'inner')

lulc_totals['percentage'] = lulc_totals.Shape_Area_x/lulc_totals.Shape_Area_y

total_check = lulc_totals.groupby(['Name']).percentage.sum()

#%% make wide dataframe

lulc_tot_wd = pd.DataFrame()

for i,site in enumerate(lulc_totals.Name.unique()):
    
    lulc_sub = lulc_totals[lulc_totals.Name==site]
    
    area = lulc_sub.Area_sq_ft.unique()[0]
    
    lulc_tot_wd = pd.concat([lulc_tot_wd,pd.DataFrame({'Site':[site],'Total Area (sq-ft)':[area]})],
                            ignore_index = True)
    
    for lulc_lev in lulc_sub.lulc.unique():
        
        perc = lulc_totals.loc[(lulc_totals.Name==site)&(lulc_totals.lulc==lulc_lev),'percentage'].values[0]
        
        lulc_tot_wd.loc[i,f'{lulc_lev} (%)'] =[perc]

#%% save output

lulc_tot_wd.to_csv(os.path.join(path_to_wqs,'Streams','outputs','LULC_watershed_percentages.csv'), index = False)
