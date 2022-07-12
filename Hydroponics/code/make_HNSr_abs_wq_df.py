# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:40:40 2021

@author: jbarrett.carter
"""
import pandas as pd
import numpy as np
import os

#%%

### Set paths and bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
abs_df_dir=os.path.join(path_to_wqs,'Data/spectra/')
wq_df_dir=os.path.join(path_to_wqs,'Hydroponics/inputs/water_quality/')
inter_dir=os.path.join(path_to_wqs,'Hydroponics/intermediates/')

wq_df_fn = 'wq_HNSr_df.csv'
abs_df_fn = 'abs_df_u2d.csv'

# Bring in data
abs_df=pd.read_csv(abs_df_dir+abs_df_fn)
wq_df=pd.read_csv(wq_df_dir+wq_df_fn)

#%%

### Some data wrangling

# Select only HNSr samples

abs_df = abs_df.loc[(abs_df.Name == 'HNSr1') | (abs_df.Name == 'HNSr2')\
                    | (abs_df.Name == 'HNSr3'),:]
abs_df = abs_df.reset_index(drop = True)
abs_df.loc[7,'Name']='HNSr1' # this sample was labelled incorrectly (probably)
# Make species names consistent

wq_df.Species.unique()

wq_df.Species[wq_df.Species=='Ammonia-Nitrogen']='Ammonium-N'
wq_df.Species[wq_df.Species=='Nitrate-Nitrogen']='Nitrate-N'

# Makes dates match

wq_dates = wq_df.Date_col.unique()
print(wq_dates)
abs_dates = abs_df.Date_col.unique()
print(abs_dates)
# wq_df['Date_col'][wq_df.Date_col=='5/3/2021']='5/4/2021'
# wq_dates = wq_df.Date_col.unique()

# Create sample IDs for combining two dataframes
# wq_cols = list(wq_df.columns[0:4])
# wq_cols.append('Name')
# wq_df.columns=wq_cols

wq_df.rename(columns={'Sample_num':'Name'},inplace=True)
wq_df['ID']=wq_df.Name+wq_df.Date_col

# concs_ind = range(int((wq_df.shape[0]-1)/3))
# concs_df = wq_df.pivot(index = range(268),columns = 'Species',values = 'Conc')

#%%

### Make dataframe with absorbances and water quality

species = wq_df.Species.unique()
species = np.append(species,'ID')
aw_df_cols = np.append(species,abs_df.columns)
abs_wq_df = abs_df

abs_wq_df.loc[:,species] = -0.1
abs_wq_df['ID']=abs_wq_df.Name+abs_wq_df.Date_col

species = np.delete(species,len(species)-1) #get rid of ID
species = np.delete(species,len(species)-1) #get rid of Conductivity

abs_wq_df = abs_wq_df[aw_df_cols]

for wq_row in range(wq_df.shape[0]):
    for abs_row in range(abs_wq_df.shape[0]):
        if wq_df.ID[wq_row]==abs_wq_df.ID[abs_row]:
            for s in species:
                if wq_df.Species[wq_row] == s:
                    abs_wq_df.loc[abs_row,s]=wq_df.Value[wq_row]

#%% Export dataframe

abs_wq_df.to_csv(inter_dir+'abs-wq_HNSr_df.csv',index=False)
