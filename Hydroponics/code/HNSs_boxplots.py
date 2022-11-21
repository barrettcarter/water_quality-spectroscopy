# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 09:25:40 2022

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
wq_path = os.path.join(path_to_wqs,'Hydroponics/inputs/water_quality/')
wq_fn = 'wq_HNSs_df.csv'
data_path = os.path.join(wq_path,wq_fn)
HNSs_df=pd.read_csv(data_path)

#%% Determine data ranges

HNSs_stats = HNSs_df.describe()

#%% Define Groups

group1 = ['NO3N','K','Ca']
group2 = ['S','P','Mg']
group3 = ['NH4N','pH','Fe']
group4 = ['Mn','B','Zn','Cu','Mo']

g1_df = HNSs_df.loc[:,group1]
g2_df = HNSs_df.loc[:,group2]
g3_df = HNSs_df.loc[:,group3]
g4_df = HNSs_df.loc[:,group4]

#%% Make plots
matplotlib.rcParams.update({'font.size': 14})
plt.figure()
plt.boxplot(g1_df,notch=True,medianprops = {'color': 'red'})
plt.ylabel('Concentration (mg/L)')
plt.xticks([1,2,3],group1)

plt.figure()
plt.boxplot(g2_df,notch=True,medianprops = {'color': 'red'})
plt.ylabel('Concentration (mg/L)')
plt.xticks([1,2,3],group2)

plt.figure()
plt.boxplot(g3_df,notch=True,medianprops = {'color': 'red'})
plt.ylabel('Concentration (mg/L)')
plt.xticks([1,2,3],group3)

plt.figure()
plt.boxplot(g4_df,notch=True,medianprops = {'color': 'red'})
plt.ylabel('Concentration (mg/L)')
plt.xticks([1,2,3,4,5],group4)

#%% make 3D plot

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(HNSs_df['Ca'],HNSs_df['K'],HNSs_df['NO3N'])
plt.xlabel('Ca (mg/L)')
plt.ylabel('K (mg/L)')
ax.set_zlabel('NO3N (mg/L)')

#%% make heatmap

HNSs_df.drop('TKN',axis = 1,inplace = True)
HNSs = HNSs_df.loc[:,'B':]
HNSs_ar = HNSs.to_numpy()
#HNSs_corrs = np.empty([HNSs.shape[1],HNSs.shape[1]])
HNSs_corrs = np.corrcoef(HNSs_ar,rowvar = False)
HNSs_map = np.empty(HNSs_corrs.shape)
#mean_sq_corr = np.mean(HNSs_corrs**2)

for r in range(HNSs_map.shape[0]):
    for c in range(HNSs_map.shape[1]):
        if r>c:
            HNSs_map[r,c]=False
        else:
            HNSs_map[r,c]=True
 
HNSs_map2 = (HNSs_map-1)**2
#HNSs_corrs2 = HNSs_corrs*HNSs_map2
HNSs_corrs2 = HNSs_corrs.flatten()
HNSs_corrs2 = HNSs_corrs2[HNSs_corrs2**2<0.99]
mean_corr = np.mean(HNSs_corrs2)
mean_sq_corr = np.mean(HNSs_corrs2**2)
sns.heatmap(HNSs_corrs,mask = HNSs_map,xticklabels = HNSs.columns,yticklabels = HNSs.columns,
            vmin = -1, vmax = 1,center =0, cmap = 'coolwarm')

