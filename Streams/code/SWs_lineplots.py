# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:29:41 2022

@author: jbarrett.carter
"""

#%% Import libraries
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
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
                     'axes.spines.top': True,
                     'figure.dpi': 300})

#%% Bring in Data

user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\' # for OneDrive
path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
wavelengths_path = os.path.join(path_to_wqs,'Data/spectra/wavelengths.csv')
spec_path1 = os.path.join(path_to_wqs,'Streams/intermediates/abs-wq_SWs_OO.csv')
SWs_df=pd.read_csv(spec_path1)
wavelengths_df = pd.read_csv(wavelengths_path)
wavelengths_df = wavelengths_df.loc[wavelengths_df.device=='Ocean Optics',:]

#%% Make dataframe for plots

SWs_abs_df = SWs_df.loc[:,'band_1':]

SWs_abs_df['Name'] = SWs_df.Name

SWs_abs_df['Storage_time'] = SWs_df.Storage_time

SWs_long_df = pd.wide_to_long(SWs_abs_df,stubnames = 'band', i=['Name','Storage_time'],j='band_num',sep='_')

SWs_long_df.reset_index(inplace=True)

SWs_long_df.rename(columns = {'band':'absorbance'},inplace=True)

SWs_long_df['band_num'] = SWs_long_df.band_num.apply(lambda x: 'band_'+str(x))

wavelengths_df.rename(columns = {'band':'band_num'},inplace = True)

wavelengths_df = wavelengths_df.loc[:,['wavelength','band_num']]

SWs_long_df = pd.merge(SWs_long_df,wavelengths_df,
                        on = 'band_num',how = 'inner')

#%% Make plots

sns.relplot(data = SWs_long_df,kind = 'line',x = 'wavelength',y='absorbance',
            col = 'Storage_time')

# sns.relplot(data = SWs_long_df.loc[SWs_long_df.Dilution==30,:],kind = 'line',x = 'wavelength',y='absorbance')

#%% Scratch

plt.figure()
plt.plot(wavelengths_df.wavelength,SWs_all_df.loc[95,'band_1':'band_1024'])
# plt.plot(wavelengths_df.wavelength,SWs_all_df.loc[98,'band_1':'band_1024'])
