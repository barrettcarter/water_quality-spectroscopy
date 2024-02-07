# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:01:15 2023

This script is for producing summary water quality plot and tables for
Streams separated by filtration.

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

#%% Set paths and make useful variables

user = os.getlogin()

path_to_wqs = r'D:\GitHub\PhD\water_quality-spectroscopy' # for external HD
# path_to_wqs = f'C:\\Users\\{user}\\Documents\\GitHub\\water_quality-spectroscopy' # for laptop

fig_dir = f'C:\\Users\\{user}\\OneDrive\\Research\\PhD\\Communications\\Images'

streams_inter_dir = os.path.join(path_to_wqs,'Streams','intermediates')

output_dir = os.path.join(path_to_wqs,'Outputs')
#%% Bring in data and concatenate

wq_strm = pd.read_csv(os.path.join(streams_inter_dir,'abs_wq_df_streams.csv'))

filtered = wq_strm.Filtered

wq_strm = wq_strm.loc[:,'Ammonium-N':'OP']

wq_strm['Filtered'] = filtered

#%% calculate summary statistics and save

for filt in [True,False]:
    
    desc_stats = wq_strm.loc[wq_strm.Filtered==filt,:].describe().T
    
    desc_stats.to_csv(os.path.join(output_dir,f'WQ_desc_stats_Filt-{filt}.csv'))

#%% make long dataframe

groups_dict = {'Nitrate-N':'Group 1', 'TKN':'Group 1', 'ON':'Group 1', 'TN':'Group 1',
               'Ammonium-N':'Group 2','Phosphate-P':'Group 2','TP':'Group 2','OP':'Group 2'}

labels_dict = {'Nitrate-N':'NO3-N','TKN':'TKN','ON':'ON','TN':'TN',
               'Ammonium-N':'NH4-N','Phosphate-P':'PO4-P','TP':'TP','OP':'OP'}

wq_long = pd.DataFrame(columns = ['Chemical Analyte','Filtration','Concentration (mg/L)',
                                  'Group'])

for species in list(labels_dict.keys()):
    
    sub_df = wq_strm.loc[:,[species,'Filtered']]
    
    sub_df.rename(columns = {species:'Concentration (mg/L)'},inplace=True)
    
    sub_df['Filtration']='NA'
    
    sub_df.loc[sub_df.Filtered==True,'Filtration']='Filtered'
    
    sub_df.loc[sub_df.Filtered==False,'Filtration']='Unfiltered'
    
    sub_df.drop(columns = 'Filtered',inplace=True)
    
    sub_df['Chemical Analyte']=labels_dict[species]
    
    sub_df['Group'] = groups_dict[species]
    
    wq_long = pd.concat([wq_long,sub_df],ignore_index=True)
    
#%% make plot

g = sns.catplot(data=wq_long,x='Chemical Analyte',y = 'Concentration (mg/L)', 
            hue = 'Filtration',col = 'Group',kind = 'box',sharex=False,sharey=False,
            palette = {'Unfiltered':'slategrey','Filtered':'lightsteelblue'})

g.set_titles(col_template="")

plt.savefig(os.path.join(fig_dir,'stream_wq_filtration_boxplot.png'),dpi = 300)
