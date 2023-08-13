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
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
wq_path = os.path.join(path_to_wqs,'Streams/inputs/water_quality/')
wq_fn = 'SWs_arl.csv'
data_path = os.path.join(wq_path,wq_fn)
SWs_df=pd.read_csv(data_path)

#%% Determine data ranges

SWs_stats = SWs_df.describe()

#%% Define Groups

group1 = ['Ca','TN']
group2 = ['Nitrate-N','TKN','ON']
group3 = ['Phosphate-P','TP']
group4 = ['OP','Ammonium-N']

g1_df = SWs_df.loc[:,group1]
g2_df = SWs_df.loc[:,group2]
g3_df = SWs_df.loc[:,group3]
g4_df = SWs_df.loc[:,group4]

#%% Make plots
# matplotlib.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = [6.5, 6.5], dpi = 300)

ax[0,0].boxplot(g1_df,notch=True,medianprops = {'color': 'red'})
ax[0,0].set_ylabel('Concentration (mg/L)')
ax[0,0].set_xticks([1,2])
ax[0,0].set_xticklabels(group1)

ax[1,0].boxplot(g2_df,notch=True,medianprops = {'color': 'red'})
ax[1,0].set_ylabel('Concentration (mg/L)')
ax[1,0].set_xticks([1,2,3])
ax[1,0].set_xticklabels(group2)

ax[0,1].boxplot(g3_df,notch=True,medianprops = {'color': 'red'})
ax[0,1].set_xticks([1,2])
ax[0,1].set_xticklabels(group3)


ax[1,1].boxplot(g4_df,notch=True,medianprops = {'color': 'red'})
ax[1,1].set_xticks([1,2])
ax[1,1].set_xticklabels(group4)


#%% make plots individually

plt.figure(dpi = 300, figsize = [3.25,3.25])
plt.boxplot(g1_df,notch=True,medianprops = {'color': 'red'})
plt.ylabel('Concentration (mg/L)')
plt.xticks([1,2],group1)

plt.figure(dpi = 300,figsize = [3.25,3.25])
plt.boxplot(g2_df,notch=True,medianprops = {'color': 'red'})
plt.ylabel('Concentration (mg/L)')
plt.xticks([1,2,3],group2)

plt.figure(dpi = 300,figsize = [3.25,3.25])
plt.boxplot(g3_df,notch=True,medianprops = {'color': 'red'})
plt.ylabel('Concentration (mg/L)')
plt.xticks([1,2],group3)

plt.figure(dpi = 300,figsize = [3.25,3.25])
plt.boxplot(g4_df,notch=True,medianprops = {'color': 'red'})
plt.ylabel('Concentration (mg/L)')
plt.xticks([1,2],group4)

#%% make 3D plot

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(SWs_df['Ca'],SWs_df['K'],SWs_df['NO3N'])
plt.xlabel('Ca (mg/L)')
plt.ylabel('K (mg/L)')
ax.set_zlabel('NO3N (mg/L)')

#%% make heatmap

# SWs_df.drop('TKN',axis = 1,inplace = True)
# SWs = SWs_df.loc[:,'B':]
SWs = SWs_df.loc[:,'Ca':]
SWs_ar = SWs.to_numpy()
#SWs_corrs = np.empty([SWs.shape[1],SWs.shape[1]])
SWs_corrs = np.corrcoef(SWs_ar,rowvar = False)
SWs_map = np.empty(SWs_corrs.shape)
#mean_sq_corr = np.mean(SWs_corrs**2)

for r in range(SWs_map.shape[0]):
    for c in range(SWs_map.shape[1]):
        if r>c:
            SWs_map[r,c]=False
        else:
            SWs_map[r,c]=True
 
SWs_map2 = (SWs_map-1)**2
#SWs_corrs2 = SWs_corrs*SWs_map2
SWs_corrs2 = SWs_corrs.flatten()
SWs_corrs2 = SWs_corrs2[SWs_corrs2**2<0.99]
mean_corr = np.mean(SWs_corrs2)
mean_sq_corr = np.mean(SWs_corrs2**2)
sns.heatmap(np.round(SWs_corrs,2),mask = SWs_map,xticklabels = SWs.columns,yticklabels = SWs.columns,
            vmin = -1, vmax = 1,center =0, cmap = 'coolwarm',annot = True)

