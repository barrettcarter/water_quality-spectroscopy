# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:56:13 2021

This code is for creating a data frame containing water solute concentrations
and corresponding uv-vis absorbances for each sample.

@author: jbarrett.carter
"""

import pandas as pd
import numpy as np
import os
import datetime as dt

#%%

def make_datetime(date_string):
    
    if '/' in date_string:
        date = dt.date(int(date_string.split(sep = '/')[2]),
                       int(date_string.split(sep = '/')[0]),
                       int(date_string.split(sep = '/')[1]))
        return(date)
    else:
        return('ERROR: Date not in right format (MM/DD/YYYY)')

#%%

### Set paths and bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
abs_df_dir=os.path.join(path_to_wqs,'Data/spectra/')
wq_df_dir=os.path.join(path_to_wqs,'Streams/inputs/water_quality/')
inter_dir = os.path.join(path_to_wqs,'Streams/intermediates/')


wq_df_fn = 'wq_streams_aj_df.csv'
wq_df_fn2 = 'wq_streams_df2.csv'
wq_codes_fn = 'ARL_codes.csv'
abs_df_fn = 'abs_df_u2d.csv'

# Bring in data
abs_df=pd.read_csv(abs_df_dir+abs_df_fn)
wq_df=pd.read_csv(wq_df_dir+wq_df_fn)
wq_df2=pd.read_csv(wq_df_dir+wq_df_fn2)
wq_codes = pd.read_csv(wq_df_dir+wq_codes_fn)

#%%

### Some data wrangling

# make ID column for wq_df and abs_df

wq_df['ID']=wq_df['Name']+wq_df['Date_col']
abs_df['ID']=abs_df['Name']+abs_df['Date_col']
abs_df = abs_df.loc[abs_df.Date_col!='11/5/2020',:]

# pivot wq_df wide

wq_df = pd.pivot_table(wq_df,values = 'Conc',columns = 'Species',index = ['ID','Name','Date_col'])
wq_df.reset_index(inplace=True)
# Select only stream samples

stream_names = wq_df['Name'].unique()

stream_abs = []

for n in abs_df.Name:
    stream_abs.append(any(stream_names==n))

abs_df = abs_df.iloc[stream_abs,:]

# fix dates (may not be needed)

# wq_df.Date_col = wq_df.Date_col.apply(lambda x: make_datetime(x))
# abs_df.Date_col = abs_df.Date_col.apply(lambda x: make_datetime(x))
# wq_df2.Date_col = wq_df2.Date_col.apply(lambda x: make_datetime(x))

# abs_dates = abs_df.Date_col.unique()
# wq_dates = wq_df.Date_col.unique()
# wq_dates2 = wq_df2.Date_col.unique()

############## OR #########################################

# abs_dates = abs_df.Date_col.apply(lambda x: make_datetime(x)).unique()
# wq_dates = wq_df.Date_col.apply(lambda x: make_datetime(x)).unique()
# wq_dates2 = wq_df2.Date_col.apply(lambda x: make_datetime(x)).unique()

############################################################

# abs_dates = np.sort(abs_dates)
# wq_dates = np.sort(wq_dates)
# wq_dates2 = np.sort(wq_dates2)
# date = abs_dates[0]

# Clean up wq_df2

wq_df2 = wq_df2.loc[pd.notna(wq_df2['ARL_code']),:]
wq_df2.reset_index(drop = True,inplace = True)

# Make new ID row (ID2) for matching up abs and wq2 data by adding in filtration

abs_df['ID2'] = abs_df.Filtered.apply(lambda x: str(x))+abs_df.ID
wq_df2['ID2'] = wq_df2.Filtered.apply(lambda x: str(x))+wq_df2.ID


#%%

### Make dataframe with absorbances and water quality

species = ['Ammonium-N','Nitrate-N','TKN','ON','TN','Phosphate-P','TP','OP']
# species = np.delete(species,0) #get rid of Ammonium
new_cols = np.concatenate((species,['ID','ID2']))
aw_df_cols = np.append(new_cols,abs_df.columns[0:1031])
abs_wq_df = abs_df

abs_wq_df.loc[:,species] = np.nan
# abs_wq_df['ID']=abs_wq_df.Name+abs_wq_df.Date_col
abs_wq_df = abs_wq_df[aw_df_cols]
abs_wq_df.reset_index(drop = True,inplace = True)

# for wq_row in range(wq_df.shape[0]):
#     for abs_row in range(abs_wq_df.shape[0]):
#         if wq_df.ID[wq_row]==abs_wq_df.ID[abs_row]:
#             for s in species:
#                 if wq_df.Species[wq_row] == s:
#                     abs_wq_df.loc[abs_row,s]=wq_df.Conc[wq_row]

for wq_ID in wq_df.ID:
    for s in species:
        if s in wq_df.columns:
            abs_wq_df.loc[abs_wq_df.ID==wq_ID,s] = float(wq_df.loc[wq_df.ID==wq_ID,s].values)

for wq_ID in wq_df2.ID2:
    for s in ['TKN','TP','Phosphate-P']:
        if s in wq_df2.columns:
            abs_wq_df.loc[abs_wq_df.ID2==wq_ID,s] = float(wq_df2.loc[wq_df2.ID2==wq_ID,s].values)

#%% Calculate ON, OP, and TN

abs_wq_df.ON = abs_wq_df.TKN-abs_wq_df['Ammonium-N']
abs_wq_df.OP = abs_wq_df.TP-abs_wq_df['Phosphate-P']
abs_wq_df.TN = abs_wq_df.TKN+abs_wq_df['Nitrate-N']


#%% Save data as csv

abs_wq_df.to_csv(inter_dir+'abs_wq_df_streams.csv')
            