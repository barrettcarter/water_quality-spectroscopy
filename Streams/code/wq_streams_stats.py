# -*- coding: utf-8 -*-
"""
Created on Wed September 22 17:45 2021

@author: J. Barrett Carter
"""
#%% bring in libraries
import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import seaborn as sns
#%% bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
wq_df_dir=os.path.join(path_to_wqs,'Streams/inputs/water_quality/')

wq_df_fn = 'wq_streams_aj_df.csv'
wq_df_fn2 = 'wq_streams_arl_df.csv'
wq_codes_fn = 'ARL_codes.csv'

# Bring in data
wq_df=pd.read_csv(wq_df_dir+wq_df_fn)
wq_df2=pd.read_csv(wq_df_dir+wq_df_fn2)
wq_codes = pd.read_csv(wq_df_dir+wq_codes_fn)


#%% some data wrangling

# clean up wq_df2
wq_df2 = wq_df2.loc[pd.notna(wq_df2['ARL_code']),:]
wq_df2.reset_index(drop = True,inplace = True)

# select only relevant codes
wq_codes = wq_codes.iloc[0:wq_df2.shape[0],:]

# create sample IDs
wq_df['ID']=wq_df['Name']+wq_df['Date_col']

wq_df2['ID']=wq_codes['Name']+wq_codes['Date_col']

# mark problematic data from 11/5/2020
# samples from 11/5 and 11/19 were given same label and stored for too long

wq_df2['ID'][wq_codes['Date_col']=='11/5/2020']=\
wq_df2['ID'][wq_codes['Date_col']=='11/5/2020']+wq_codes['Date_an'][wq_codes['Date_col']=='11/5/2020']


# re-order wq_df to match wq_df2

# separate filtered and unfiltered samples

wq_df2_fil = wq_df2.loc[wq_codes.Filtered==True,:]
wq_df2_unf = wq_df2.loc[wq_codes.Filtered==False,:]

# add ARL codes to wq_df separated by filtration
# row = 5
for row in range(wq_df.shape[0]):

    fil_code = wq_df2_fil.loc[wq_df2_fil['ID']==wq_df['ID'][row],'ARL_code'].values
    if fil_code.size != 1:
        wq_df.loc[row,'Fil_code'] = np.nan
        
    else:
        wq_df.loc[row,'Fil_code']=fil_code
        
    unf_code = wq_df2_unf.loc[wq_df2_unf['ID']==wq_df['ID'][row],'ARL_code'].values
    if unf_code.size != 1:
        wq_df.loc[row,'Unf_code'] = np.nan
        
    else:
        wq_df.loc[row,'Unf_code']=unf_code
        
wq_df = wq_df.loc[np.isnan(wq_df['Fil_code'])==False,:]

# pivot wq_df to make separate species columns

conc_df = pd.pivot_table(wq_df,values = 'Conc',columns = ['Species'],index = ['ID'])
conc_df.reset_index(inplace=True)
conc_df_fil = pd.pivot_table(wq_df,values = 'Conc',columns = ['Species'],index = ['Fil_code'])
conc_df_fil.reset_index(inplace=True)
conc_df_unf = pd.pivot_table(wq_df,values = 'Conc',columns = ['Species'],index = ['Unf_code'])
conc_df_unf.reset_index(inplace=True)

#%% make subsets

Nit_fil = pd.DataFrame(columns = ['Lab1','Lab2'])
Nit_fil['Lab1']=conc_df_fil['Nitrate-N']
Nit_fil['Lab2']=wq_df2.loc[wq_df2['ARL_code'].isin(conc_df_fil['Fil_code']),'Nitrate-N'].values
Nit_fil['Lab2']=pd.to_numeric(Nit_fil['Lab2'])
#%% make plots
sns.set_theme(font_scale = 1.25,style='ticks')
plt.plot(Nit_fil['Lab1'],Nit_fil['Lab2'],'k.')
plt.plot([0,3],[0,3],'--k')