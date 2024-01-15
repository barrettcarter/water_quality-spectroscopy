# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:01:15 2023

This script is for producing summary water quality plot and tables for HNS and
Streams together.

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

hns_inter_dir = os.path.join(path_to_wqs,'Hydroponics','intermediates')

streams_inter_dir = os.path.join(path_to_wqs,'Streams','intermediates')

output_dir = os.path.join(path_to_wqs,'Outputs')
#%% Bring in data and concatenate

wq_hns = pd.read_csv(os.path.join(hns_inter_dir,'abs-wq_HNSr_df.csv'))

wq_hns = wq_hns.loc[:,'Nitrate-N':'pH']

for col in wq_hns.columns:

    wq_hns.loc[wq_hns[col]<0,col] = np.nan


wq_strm = pd.read_csv(os.path.join(streams_inter_dir,'abs_wq_df_streams.csv'))

wq_strm = wq_strm.loc[:,'Ammonium-N':'OP']

wq_dict = {'HNS':wq_hns,'stream':wq_strm}

sample_types = list(wq_dict.keys())

#%% calculate summary statistics and save

stype = sample_types[1] # for testing

for stype in sample_types:
    
    desc_stats = wq_dict[stype].describe().T
    
    desc_stats.to_csv(os.path.join(output_dir,f'WQ_desc_stats_{stype}.csv'))

#%% make plot

groups_dict = {'g1':['Nitrate-N','Potassium','Calcium','Sulfate'],
               'g2':['Phosphorus','Magnesium'],
               'g3':['Ammonium-N','pH','Iron'],
               'g4':['Manganese','Boron','Zinc','Copper','Molybdenum'],
               'g5':['Nitrate-N','TKN','ON','TN'],
               'g6':['Ammonium-N','Phosphate-P','TP','OP']}

group_stype = {'g1':'HNS','g2':'HNS','g3':'HNS','g4':'HNS','g5':'stream','g6':'stream'}

group_subplot = {'g1':'fig.add_subplot(5,5,(1,4))',
                 'g2':'fig.add_subplot(5,5,(6,7))',
                 'g3':'fig.add_subplot(5,5,(8,10))',
                 'g4':'fig.add_subplot(5,5,(11,15))',
                 'g5':'fig.add_subplot(5,5,(16,19))',
                 'g6':'fig.add_subplot(5,5,(21,24))'}

labels_df = pd.DataFrame({'name':['Nitrate-N','Potassium','Calcium','Sulfate',
                                  'Phosphorus','Magnesium','Ammonium-N','pH',
                                  'Iron','Manganese','Boron','Zinc','Copper',
                                  'Molybdenum','TKN','ON','TN','Phosphate-P',
                                  'TP','OP'],
                          'label':['NO3-N','K','Ca','S','P','Mg','NH4-N','pH',
                                   'Fe','Mn','B','Zn','Cu','Mb','TKN','ON','TN',
                                   'PO4-P','TP','OP']})

stype_colors = {'HNS':'tab:red','stream':'tab:blue'}

fig = plt.figure(dpi = 300,figsize = [6.5,6.5],tight_layout = True)

plt.figtext(0.5,0,'Chemical Analyte',ha = 'center',va = 'center')

plt.figtext(0,0.5,'Concentration (mg/L; unitless for pH)',rotation = 'vertical',
         va = 'center',ha = 'center')

group = 'g1' # for testing

for group in list(groups_dict.keys()):
    
    analytes = groups_dict[group]
    
    stype = group_stype[group]
    
    wq_df = wq_dict[stype]
    
    labels = labels_df.loc[labels_df.name.isin(analytes),'label']
    
    stype_color = stype_colors[stype]
    
    ax = eval(group_subplot[group])
    
    # ax.boxplot(wq_df.loc[:,analytes],notch=True,boxprops={'color': stype_color})
    sns.boxplot(wq_df.loc[:,analytes],color = stype_color)
    
    ax.set_xticks(np.arange(0,len(analytes)))
    ax.set_xticklabels(labels)
    
    # ax.set_title(stype)