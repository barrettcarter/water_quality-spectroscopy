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

#%% Bring in Data

user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\' # for OneDrive
path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
wavelengths_path = os.path.join(path_to_wqs,'Data/spectra/wavelengths.csv')
spec_path1 = os.path.join(path_to_wqs,'Hydroponics/intermediates/abs-wq_HNSr_df.csv')
spec_path2 = os.path.join(path_to_wqs,'Hydroponics/intermediates/abs-wq_HNSrd30_df.csv')
HNSr_df=pd.read_csv(spec_path1)
HNSrd30_df = pd.read_csv(spec_path2)
wavelengths_df = pd.read_csv(wavelengths_path)
wavelengths_df = wavelengths_df.loc[wavelengths_df.device=='Ocean Optics',:]

HNSr_df = HNSr_df.loc[0:62,:]
HNSrd30_df = HNSrd30_df.loc[0:57,:]

#%% Make dataframe for plots

HNSr_df['Dilution'] = 1

HNSrd30_df['Dilution'] = 30

HNSr_all_df = pd.concat([HNSr_df,HNSrd30_df],ignore_index = True)

HNSr_all_df.drop(index = 98,inplace = True) # remove duplicate

HNSr_long_df = pd.wide_to_long(HNSr_all_df,stubnames = 'band', i='ID',j='band_num',sep='_')

HNSr_long_df.reset_index(inplace=True)

HNSr_long_df.rename(columns = {'band':'absorbance'},inplace=True)

HNSr_long_df['band_num'] = HNSr_long_df.band_num.apply(lambda x: 'band_'+str(x))

wavelengths_df.rename(columns = {'band':'band_num'},inplace = True)

wavelengths_df = wavelengths_df.loc[:,['wavelength','band_num']]

HNSr_long_df = pd.merge(HNSr_long_df,wavelengths_df,
                        on = 'band_num',how = 'inner')

#%% Make plots

sns.relplot(data = HNSr_long_df,kind = 'line',x = 'wavelength',y='absorbance',
            col = 'Dilution')

sns.relplot(data = HNSr_long_df.loc[HNSr_long_df.Dilution==30,:],kind = 'line',x = 'wavelength',y='absorbance')

#%% Scratch

plt.figure()
plt.plot(wavelengths_df.wavelength,HNSr_all_df.loc[95,'band_1':'band_1024'])
# plt.plot(wavelengths_df.wavelength,HNSr_all_df.loc[98,'band_1':'band_1024'])
