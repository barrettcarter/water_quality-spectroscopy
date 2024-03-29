# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:01:15 2023

This script is for producing summary water quality plot and tables for
Streams and HNS (diluted and undiluted) samples in comparison to synthetic samples.

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

inter_dirs = {'Streams':os.path.join(path_to_wqs,'Streams','intermediates'),
              'HNS':os.path.join(path_to_wqs,'Hydroponics','intermediates')}

output_dirs = {'Streams':os.path.join(path_to_wqs,'Streams','outputs'),
              'HNS':os.path.join(path_to_wqs,'Hydroponics','outputs')}

real_fns = {'Streams':'abs_wq_df_streams.csv',
            'HNS':'abs-wq_HNSr_df.csv'}

syn_fns = {'Streams':'abs-wq_SWs_OO.csv',
            'HNS':'abs-wq_HNSs_df.csv'}

stypes = list(inter_dirs.keys())

# groups_dict = {'Nitrate-N':'Group 1', 'TKN':'Group 1', 'ON':'Group 1', 'TN':'Group 1',
#                'Ammonium-N':'Group 2','Phosphate-P':'Group 2','TP':'Group 2','OP':'Group 2'}

# labels_dict = {'Nitrate-N':'NO3-N','TKN':'TKN','ON':'ON','TN':'TN',
#                'Ammonium-N':'NH4-N','Phosphate-P':'PO4-P','TP':'TP','OP':'OP'}

# plot_info = pd.DataFrame({'sample_type':['stream','stream','stream','stream',
#                                          'stream','stream','stream','stream',
#                                          'HNS','HNS','HNS','HNS','HNS','HNS','HNS',
#                                          'HNS','HNS','HNS','HNS','HNS','HNS','HNS'],
#                           'analyte':['Nitrate-N', 'TKN', 'ON', 'TN',
#                                      'Ammonium-N','Phosphate-P','TP','OP',
#                                      'Nitrate-N','Potassium','Calcium','Sulfate',
#                                      'Phosphorus','Magnesium','Ammonium-N','pH',
#                                      'Iron','Manganese','Boron','Zinc','Copper','Molybdenum'],
#                           'group':['Group 1','Group 1','Group 1','Group 1',
#                                    'Group 2','Group 2','Group 2','Group 2',
#                                    'Group 1','Group 1','Group 1','Group 1',
#                                    'Group 2','Group 2','Group 3'],
#                           'label':[]})

# 'g1':['Nitrate-N','Potassium','Calcium','Sulfate'],
#                'g2':['Phosphorus','Magnesium'],
#                'g3':['Ammonium-N','pH','Iron'],
#                'g4':['Manganese','Boron','Zinc','Copper','Molybdenum']

#%% make plots for each sample type

desc_stats_df = pd.DataFrame()

wq_dict = {}

stype = stypes[1] # for testing

for stype in stypes:
    
    inter_dir = inter_dirs[stype]
    
    output_dir = output_dirs[stype]
    
    real_fn = real_fns[stype]
    
    syn_fn = syn_fns[stype]
    
    
    ### Bring in data and concatenate

    real_df = pd.read_csv(os.path.join(inter_dir,real_fn))
    
    syn_df = pd.read_csv(os.path.join(inter_dir,syn_fn))
    
    if stype=='Streams':
    
        real_df = real_df.loc[real_df.Filtered,'Ammonium-N':'OP']
    
        syn_df = syn_df.loc[syn_df.Storage_time==10,'Ammonium-N':'TN']
        
    else:
        
        real_df = real_df.loc[:,'Nitrate-N':'pH']
        
        syn_df = syn_df.loc[:,'pH':'Nitrate-N']
        
    
    real_df['origin'] = 'natural'
    
    syn_df['origin'] = 'synthetic'
    
    all_df = pd.concat([real_df,syn_df])
    
    all_long = pd.DataFrame()
    
    for col in all_df.columns:
        
        if col != 'origin':
            
            sub_df = all_df.loc[:,[col,'origin']]
            
            sub_df.rename(columns = {col:'concentration'},inplace=True)
            
            sub_df['analyte'] = col
            
            all_long = pd.concat([all_long,sub_df],ignore_index = True)
            
    wq_dict[stype] = all_long
    
    ### calculate summary statistics and compile into dataframe
    
    desc_stats = all_long.groupby(['origin','analyte']).concentration.describe().reset_index()
    
    desc_stats['sample_type'] = stype
    
    desc_stats_df = pd.concat([desc_stats_df,desc_stats],ignore_index=True)
    
#%% save decriptive stats

output_dir = os.path.join(path_to_wqs,'Outputs')
    
desc_stats_df.to_csv(os.path.join(output_dir,'WQ_desc_stats_syn-aug.csv'),index=False)

#%% make plot

groups_dict = {'g1':['Nitrate-N','Potassium','Calcium','Sulfate'],
               'g2':['Phosphorus','Magnesium'],
               'g3':['Ammonium-N','pH','Iron'],
               'g4':['Manganese','Boron','Zinc','Copper','Molybdenum'],
               'g5':['Nitrate-N','TKN','ON','TN'],
               'g6':['Ammonium-N','Phosphate-P','TP','OP']}

group_stype = {'g1':'HNS','g2':'HNS','g3':'HNS','g4':'HNS','g5':'Streams','g6':'Streams'}

# group_subplot = {'g1':'fig.add_subplot(5,5,(1,4))',
#                  'g2':'fig.add_subplot(5,5,(6,7))',
#                  'g3':'fig.add_subplot(5,5,(8,10))',
#                  'g4':'fig.add_subplot(5,5,(11,15))',
#                  'g5':'fig.add_subplot(5,5,(16,19))',
#                  'g6':'fig.add_subplot(5,5,(21,24))'}

group_subplot = {'g1':'fig.add_subplot(3,16,(1,9))',
                 'g2':'fig.add_subplot(3,16,(17,21))',
                 'g3':'fig.add_subplot(3,16,(11,16))',
                 'g4':'fig.add_subplot(3,16,(23,32))',
                 'g5':'fig.add_subplot(3,32,(65,79))',
                 'g6':'fig.add_subplot(3,32,(82,96))'}

labels_df = pd.DataFrame({'name':['Nitrate-N','Potassium','Calcium','Sulfate',
                                  'Phosphorus','Magnesium','Ammonium-N','pH',
                                  'Iron','Manganese','Boron','Zinc','Copper',
                                  'Molybdenum','TKN','ON','TN','Phosphate-P',
                                  'TP','OP'],
                          'label':['NO3-N','K','Ca','SO4','P','Mg','NH4-N','pH',
                                   'Fe','Mn','B','Zn','Cu','Mb','TKN','ON','TN',
                                   'PO4-P','TP','OP']})

# stype_colors = {'HNS':'tab:red','stream':'tab:blue'}

fig = plt.figure(dpi = 300,figsize = [10,10],constrained_layout = True)

plt.figtext(0.5,0.08,'Chemical Analyte',ha = 'center',va = 'center')

plt.figtext(0.08,0.5,'Concentration (mg/L)',rotation = 'vertical',
         va = 'center',ha = 'center')

group = 'g1' # for testing

for group in list(groups_dict.keys()):
    
    analytes = groups_dict[group]
    
    stype = group_stype[group]
    
    wq_df = wq_dict[stype]
    
    wq_df = wq_df.loc[(wq_df.analyte.isin(analytes)),:]
    
    # labels = labels_df.loc[labels_df.name.isin(analytes),'label']
    
    wq_df = pd.merge(wq_df,labels_df,left_on='analyte',right_on='name',how='inner')
    
    # stype_color = stype_colors[stype]
    
    ax = eval(group_subplot[group])
    
    # ax.boxplot(wq_df.loc[:,analytes],notch=True,boxprops={'color': stype_color})
    sns.boxplot(data=wq_df,x='label',y='concentration',hue = 'origin')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(stype)
    
    # ax.set_xticks(np.arange(0,len(analytes)))
    # ax.set_xticklabels(labels)
    
    # ax.set_title(stype)

#%% make useful variables for heatmaps

wq_strm.rename(columns = labels_dict,inplace=True)

wq_strm.loc[wq_strm.Filtered,'Filtration'] = 'Filtered'
wq_strm.loc[wq_strm.Filtered==False,'Filtration'] = 'Unfiltered'

filtration = wq_strm.Filtration.unique()

#%% make heatmaps

filt = 'Unfiltered' # for testing

wq_corrs_dict = {}

for filt in filtration:
    
    wq_df = wq_strm.loc[wq_strm.Filtration==filt,'NH4-N':'OP']
    wq = wq_df[wq_df.isna().any(axis=1)==False]
    
    wq_ar = wq.to_numpy()
    wq_corrs = np.corrcoef(wq_ar,rowvar = False)
    
    wq_corrs_dict[filt] = wq_corrs
    
    wq_map = np.empty(wq_corrs.shape)
    
    for r in range(wq_map.shape[0]):
        for c in range(wq_map.shape[1]):
            if r>c:
                wq_map[r,c]=False
            else:
                wq_map[r,c]=True
     
    plt.figure(dpi = 300)
    
    plt.title(filt)
    
    sns.heatmap(wq_corrs,mask = wq_map,xticklabels = wq.columns,yticklabels = wq.columns,
                vmin = -1, vmax = 1,center =0, cmap = 'coolwarm',
                annot = True,annot_kws={'size':'x-small'},fmt = '.2f')
    
    # plt.savefig(os.path.join(fig_dir,f'{filt}_WQ_heatmap.png'),dpi=300,bbox_inches='tight')
    
#%% make difference heatmap

wq_corrs_filt = wq_corrs_dict['Filtered']
wq_corrs_unf = wq_corrs_dict['Unfiltered']

wq_corrs_dif = wq_corrs_filt - wq_corrs_unf

mean_corr_dif = np.mean(wq_corrs_dif)

# wq_corrs_pdif = np.round(wq_corrs_dif/wq_corrs_unf*100).astype(int)

wq_corrs_pdif = wq_corrs_dif/np.abs(wq_corrs_unf)

diff_dict = {'Difference':wq_corrs_dif,'Percent Difference': wq_corrs_pdif}

fmt_dict = {'Difference':'.2f','Percent Difference':'.0%'}

for diff in list(diff_dict.keys()):
    
    wq_corrs = diff_dict[diff]
    
    plt.figure(dpi = 300)
    
    plt.title(diff)
    
    sns.heatmap(wq_corrs,mask = wq_map,xticklabels = wq.columns,yticklabels = wq.columns,
                center =0, cmap = 'coolwarm',
                annot = True,annot_kws={'size':'x-small'},fmt = fmt_dict[diff])
    
    plt.savefig(os.path.join(fig_dir,f'{diff}_WQ_heatmap.png'),dpi=300,bbox_inches='tight')
    
#%% r-squared heatmaps

wq_rsq_filt = wq_corrs_filt**2
wq_rsq_unf = wq_corrs_unf**2

wq_rsq_dict = {'Filtered':wq_rsq_filt,'Unfiltered':wq_rsq_unf}

for filtration in ['Filtered','Unfiltered']:
    
    wq_corrs = wq_rsq_dict[filtration]
    
    wq_map = np.empty(wq_corrs.shape)
    
    for r in range(wq_map.shape[0]):
        for c in range(wq_map.shape[1]):
            if r>c:
                wq_map[r,c]=False
            else:
                wq_map[r,c]=True
     
    plt.figure(dpi = 300)
    
    plt.title(f'$r^2$ {filtration}')
    
    sns.heatmap(wq_corrs,mask = wq_map,xticklabels = wq.columns,yticklabels = wq.columns,
                vmin = 0, vmax = 1,center =0.5, cmap = 'Reds',
                annot = True,annot_kws={'size':'x-small'},fmt = '.3f')
    
    plt.savefig(os.path.join(fig_dir,f'{filtration}_WQ_rsq_heatmap.png'),dpi=300,bbox_inches='tight')
    
#%% make difference heatmap

wq_rsq_dif = wq_rsq_filt - wq_rsq_unf

mean_rsq_fil = np.mean(wq_rsq_filt)

mean_rsq_unf = np.mean(wq_rsq_unf)

mean_rsq_dif = np.mean(wq_rsq_dif)

# wq_rsq_pdif = np.round(wq_rsq_dif/wq_rsq_unf*100).astype(int)

wq_rsq_pdif = wq_rsq_dif/np.abs(wq_rsq_unf)

diff_dict = {'Difference':wq_rsq_dif,'Percent Difference': wq_rsq_pdif}

fmt_dict = {'Difference':'.2f','Percent Difference':'.0%'}

for diff in list(diff_dict.keys()):
    
    wq_rsq = diff_dict[diff]
    
    plt.figure(dpi = 300)
    
    plt.title(diff)
    
    sns.heatmap(wq_rsq,mask = wq_map,xticklabels = wq.columns,yticklabels = wq.columns,
                center =0, cmap = 'coolwarm',
                annot = True,annot_kws={'size':'x-small'},fmt = fmt_dict[diff])
    
    plt.savefig(os.path.join(fig_dir,f'{diff}_WQ_rsq_heatmap.png'),dpi=300,bbox_inches='tight')
