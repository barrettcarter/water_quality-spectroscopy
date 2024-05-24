# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:01:15 2023

This script is for producing summary absorbance spectra for plots for HNS and
Streams, separated by origin (natural versus synthetic), in a single figure.

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

abs_df = pd.read_csv(os.path.join(hns_inter_dir,'abs-wq_HNSr_df.csv'))

abs_df = abs_df.loc[:,'band_1':]

abs_df['sample_type'] = 'natural, undiluted HNS'

files = [os.path.join(hns_inter_dir,'abs-wq_HNSs_df.csv'),
         os.path.join(hns_inter_dir,'abs-wq_HNSrd30_df.csv'),
         os.path.join(hns_inter_dir,'abs-wq_HNSsd30_df.csv'),
         os.path.join(streams_inter_dir,'abs_wq_df_streams.csv'),
         os.path.join(streams_inter_dir,'abs-wq_SWs_OO.csv')]

labels = ['synthetic, undiluted HNS','natural, diluted HNS',
          'synthetic, diluted HNS','natural stream', 'synthetic stream']

for file,label in zip(files,labels):
    
    abs_sub = pd.read_csv(file)
    
    if label == 'natural stream':
    
        abs_sub = abs_sub.loc[abs_sub.Filtered==True,:]
        
    abs_sub['sample_type'] = label
    
    abs_sub = abs_sub.loc[:,'band_1':'sample_type']

    abs_df = pd.concat([abs_df,abs_sub],ignore_index=True)

abs_med = abs_df.groupby('sample_type').median().T

abs_av = abs_df.groupby('sample_type').mean().T

abs_sd = abs_df.groupby('sample_type').std().T

abs_q1 = abs_df.groupby('sample_type').quantile(0.25).T

abs_q3 = abs_df.groupby('sample_type').quantile(0.75).T

wavelengths = pd.read_csv(os.path.join(path_to_wqs,'Data','spectra','wavelengths.csv'))

wavelengths = wavelengths.loc[wavelengths.device=='Ocean Optics','wavelength']

#%% make plot

sample_types = abs_df.sample_type.unique()

fig, axs = plt.subplots(nrows = 3, ncols = 2, sharey = False, dpi = 300,
                        figsize = [6.5,8.5], tight_layout = True)

axs = fig.axes

fig.text(0.5,0,'wavelength (nm)',ha = 'center',va = 'top')
fig.text(0,0.5,'absorbance (unitless)',va = 'center',ha = 'right',rotation = 'vertical')

iax = 0

for stype in sample_types:
    
    ax = axs[iax]
    
    ### use median, q1, and q3
    
    ax.fill_between(wavelengths,abs_q3[stype],abs_q1[stype],color = 'lightskyblue')
    
    ax.plot(wavelengths,abs_med[stype],color = 'dodgerblue')
    
    ### OR use av +/- 1 sd
    
    # ax.fill_between(wavelengths,abs_av[stype]+abs_sd[stype],abs_av[stype]-abs_sd[stype],
    #                 color = 'lightskyblue')
    
    # ax.plot(wavelengths,abs_av[stype],color = 'dodgerblue')
    
    ax.set_title(stype)
    
    iax +=1
        
#%% make some statistics

abs_med_stats = abs_med.describe().T

abs_av_stats = abs_av.describe().T.reset_index(names = ['Sample Type - Treatment'])

abs_av_stats['Statistic'] = 'mean'

abs_sd_stats = abs_sd.describe().T.reset_index(names = ['Sample Type - Treatment'])

abs_sd_stats['Statistic'] = 'SD'

abs_stats = pd.concat([abs_av_stats,abs_sd_stats],ignore_index = True)

#%% save statistics

abs_stats.to_csv(os.path.join(output_dir,'abs_stats_syn-aug.csv'),index = False)

#%% determine max absorbances

abs_av.reset_index(inplace = True, names = 'waveband')
abs_av['wavelengths'] = wavelengths

abs_av_max = abs_av.iloc[:,1:-1].max()

#%% determine peak wavelengths

abs_peaks = pd.DataFrame()

for stype in abs_av.columns[1:-1]:
    
    # print(abs_av.loc[abs_av[stype]==abs_av_max[stype],'wavelengths'])
    
    abs_peaks.loc[0,stype] = abs_av.loc[abs_av[stype]==abs_av_max[stype],'wavelengths'].values[0]
    
abs_peaks.loc[1,:] = abs_av_max

abs_peaks.rename(index = {0:'peak wavelength',1:'peak absorbance'},inplace=True)

#%% save peak wavelengths

abs_peaks.to_csv(os.path.join(output_dir,'peak wavelengths - syn-aug.csv'))
