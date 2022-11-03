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
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
spec_path1 = os.path.join(path_to_wqs,'Data/spectra/abs_df_u2d.csv')
spec_path2 = os.path.join(path_to_wqs,'Hydroponics/inputs/absorbance/abs_HNSs_nanodrop_all_trans.csv')
HNSs_gs_df=pd.read_csv(spec_path1)
HNSs_nd_df = pd.read_csv(spec_path2)

#%% Some usefult variables

sample_nums = np.linspace(1,46,46,dtype = int)
sam_num_df = pd.DataFrame(sample_nums,columns = ['number'])

def add_HNSs(num):
    name = 'HNSs'+str(num)
    return(name)

sam_num_df['Nameconc']=sam_num_df['number'].apply(add_HNSs)
sam_num_df['Named30']=sam_num_df['Nameconc']+'d30'
sam_names_all = pd.wide_to_long(sam_num_df,['Name'],i='number',j='dilution',suffix=r'\w+')

#%% Make dataframes for plots

HNSs_gs_df.insert(1,'ID',HNSs_gs_df['Name']+HNSs_gs_df['Date_an']+HNSs_gs_df['Time_an'])

# GatorSpec - concentrated

HNSs_gs = HNSs_gs_df.loc[HNSs_gs_df['Name'].isin(sam_num_df.loc[:,'Nameconc']),:]
HNSs_gs_long=pd.wide_to_long(HNSs_gs,'band',i='ID',j='band_num',sep='_')
HNSs_gs_long.rename(columns={'band':'absorbance'},inplace=True)
HNSs_gs_long.reset_index(inplace=True)
# band_nums = HNSs_gs_long['band_num']
# wavelengths = 190+band_nums*(band_nums-1)(667-190)/(1023)
HNSs_gs_long.insert(2,'wavelength (nm)',190+(HNSs_gs_long['band_num']-1)*(667-190)/(1023))

# GatorSpec - diluted

HNSs_gs_d30 = HNSs_gs_df.loc[HNSs_gs_df['Name'].isin(sam_num_df.loc[:,'Named30']),:]
HNSs_gs_d30_long=pd.wide_to_long(HNSs_gs_d30,'band',i='ID',j='band_num',sep='_')
HNSs_gs_d30_long.rename(columns={'band':'absorbance'},inplace=True)
HNSs_gs_d30_long.reset_index(inplace=True)
# band_nums = HNSs_gs_long['band_num']
# wavelengths = 190+band_nums*(band_nums-1)(667-190)/(1023)
HNSs_gs_d30_long.insert(2,'wavelength (nm)',190+(HNSs_gs_d30_long['band_num']-1)*(667-190)/(1023))

# NanoDrop

# HNSs_nd_cols = HNSs_nd_df.columns.to_numpy()
# HNSs_nd_cols = pd.Series(HNSs_nd_cols)
# keep_col = HNSs_nd_cols.isin(sam_num_df.loc[:,'Nameconc'])

HNSs_nd_long = pd.wide_to_long(HNSs_nd_df,[' HNSs'],i='Wavelength (nm)',j='sample number',suffix=r'\w+')
HNSs_nd_long.reset_index(inplace=True)
HNSs_nd_long = HNSs_nd_long.loc[:,['Wavelength (nm)','sample number', ' HNSs']]
HNSs_nd_long.rename(columns={'Wavelength (nm)':'wavelength (nm)',' HNSs':'absorbance'},inplace = True)

# get diluted

HNSs_nd_long['sample number']= 'HNSs'+ HNSs_nd_long['sample number']
HNSs_nd_d30_long = HNSs_nd_long.loc[HNSs_nd_long['sample number'].isin(sam_num_df['Named30']),:]

# keep concentrated

HNSs_nd_long = HNSs_nd_long.loc[HNSs_nd_long['sample number'].isin(sam_num_df['Nameconc']),:]

#%% Make plots

sns.relplot(data = HNSs_gs_long,kind = 'line',x = 'wavelength (nm)',y='absorbance')
sns.relplot(data = HNSs_gs_d30_long,kind = 'line',x = 'wavelength (nm)',y='absorbance')
sns.relplot(data = HNSs_nd_long,kind = 'line',x = 'wavelength (nm)',y='absorbance')
sns.relplot(data = HNSs_nd_d30_long,kind = 'line',x = 'wavelength (nm)',y='absorbance')
