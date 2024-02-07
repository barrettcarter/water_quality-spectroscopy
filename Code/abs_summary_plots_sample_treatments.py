# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:01:15 2023

This script is for producing summary absorbance spectra for plots for HNS and
Streams, separated by sample treatment, in a single figure.

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
#%% Bring in data and concatenate

abs_df = pd.read_csv(os.path.join(hns_inter_dir,'abs-wq_HNSr_df.csv'))

abs_df = abs_df.loc[:,'band_1':]

abs_df['sample_type'] = 'HNS - Undiluted'

files = [os.path.join(hns_inter_dir,'abs-wq_HNSrd30_df.csv'),
         os.path.join(streams_inter_dir,'abs_wq_df_streams.csv')]

labels = ['HNS - Diluted','Stream']

for file,label in zip(files,labels):
    
    abs_sub = pd.read_csv(file)
    
    if label == labels[0]:
    
        abs_sub['sample_type'] = labels[0]
        
    else:
        
        abs_sub.loc[abs_sub.Filtered==True,'sample_type'] = 'Stream - Filtered'
        abs_sub.loc[abs_sub.Filtered==False,'sample_type'] = 'Stream - Unfiltered'
    
    abs_sub = abs_sub.loc[:,'band_1':'sample_type']

    abs_df = pd.concat([abs_df,abs_sub])

abs_med = abs_df.groupby('sample_type').median().T

abs_q1 = abs_df.groupby('sample_type').quantile(0.25).T

abs_q3 = abs_df.groupby('sample_type').quantile(0.75).T

wavelengths = pd.read_csv(os.path.join(path_to_wqs,'Data','spectra','wavelengths.csv'))

wavelengths = wavelengths.loc[wavelengths.device=='Ocean Optics','wavelength']

#%% make plot

sample_types = abs_df.sample_type.unique()

fig, axs = plt.subplots(nrows = 2, ncols = 2, sharey = True, dpi = 300,
                        figsize = [6.5,6.5], tight_layout = True)

axs = fig.axes

fig.text(0.5,0,'wavelength (nm)',ha = 'center',va = 'top')
fig.text(0,0.5,'absorbance (unitless)',va = 'center',ha = 'right',rotation = 'vertical')

iax = 0

for stype in sample_types:
    
    ax = axs[iax]
    
    ax.fill_between(wavelengths,abs_q3[stype],abs_q1[stype],color = 'lightskyblue')
    
    ax.plot(wavelengths,abs_med[stype],color = 'dodgerblue')
    
    ax.set_title(stype)
    
    iax +=1