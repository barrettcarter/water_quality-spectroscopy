# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:40:40 2021

@author: jbarrett.carter
"""
#%% bring in libraries
import pandas as pd
import os
# import datetime as dt
# import scipy
#%% bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
wq_df_dir=os.path.join(path_to_wqs,'Streams/inputs/water_quality/')

wq_df_fn2 = 'wq_streams_arl_df.csv'
wq_codes_fn = 'ARL_codes.csv'

# Bring in data

wq_df2=pd.read_csv(wq_df_dir+wq_df_fn2)
wq_codes = pd.read_csv(wq_df_dir+wq_codes_fn)

#%% some data wrangling

# clean up wq_df2
wq_df2 = wq_df2.loc[pd.notna(wq_df2['ARL_code']),:]
wq_df2.reset_index(drop = True,inplace = True)

# c = wq_df2.columns[1]
for c in wq_df2.columns:
    wq_df2.loc[:,c] = pd.to_numeric(wq_df2[c])

# Make units consistent (mg/L for all)

wq_df2.loc[:,'Phosphate-P'] = wq_df2.loc[:,'Phosphate-P']/1000
wq_df2.loc[:,'TP'] = wq_df2.loc[:,'TP']/1000

# select only relevant codes
wq_codes = wq_codes.iloc[0:wq_df2.shape[0],:]

# create sample Name and ID columns

wq_df2['Name'] = wq_codes['Name']
wq_df2['Date_col']=wq_codes['Date_col']
wq_df2['Filtered']=wq_codes['Filtered']
wq_df2['ID']=wq_codes['Name']+wq_codes['Date_col']

# mark problematic data from 11/5/2020
# samples from 11/5 and 11/19 were given same label and stored for too long

wq_df2['ID'][wq_codes['Date_col']=='11/5/2020']=\
wq_df2['ID'][wq_codes['Date_col']=='11/5/2020']+wq_codes['Date_an'][wq_codes['Date_col']=='11/5/2020']

#%% Export dataframe

wq_df2.to_csv(wq_df_dir+'wq_streams_df2.csv',index=False)
