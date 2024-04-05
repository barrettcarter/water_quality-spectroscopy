# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:40:40 2021

@author: jbarrett.carter
"""
import pandas as pd
# import numpy as np
import os

#%% Set paths and bring in data

user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
path_to_wqs = r'D:\GitHub\PhD\water_quality-spectroscopy' # for external HD
abs_df_dir=os.path.join(path_to_wqs,'Data/spectra/')
wq_df_dir=os.path.join(path_to_wqs,'Hydroponics/inputs/water_quality/')
inter_dir=os.path.join(path_to_wqs,'Hydroponics/intermediates/')

wq_df_fn = 'wq_HNSs_df.csv'
abs_df_fn = 'abs_df_u2d.csv'

# Bring in data
abs_df=pd.read_csv(abs_df_dir+abs_df_fn)
wq_df=pd.read_csv(wq_df_dir+wq_df_fn)

#%% Some data wrangling

# Add 'Name' to wq_df specifically for HNSsd30 samples
wq_df['Name']=wq_df['ID#'].apply(lambda x: 'HNSs'+str(x))
#HNSs_inds = abs_df['Name'].apply(lambda x: x in wq_df['Name'])

# Select only HNSs samples
HNSs_abs = abs_df.loc[abs_df['Name'].isin(wq_df['Name']),:].reset_index(drop = True)

# get rid of duplicates
HNSs_abs = HNSs_abs.loc[4:,:]

HNSs_abs.reset_index(inplace = True, drop = True)

#%% Make dataframe with absorbances and water quality
# abs_row = 0 #for testing
# species = wq_df.columns[2:18]
# abs_wq_df = HNSs_abs.copy()
# abs_wq_df[species]=-0.1

# for abs_row in range(HNSs_abs.shape[0]):
#     wq_row = wq_df.loc[wq_df.Name == HNSs_abs['Name'][abs_row],'pH':'NO3N']
#     abs_wq_df.loc[abs_row,species]=wq_row.values[0]
    
# cols = list(species)
# cols2 = list(HNSs_abs.columns)
# cols = cols+cols2

# abs_wq_df = abs_wq_df.loc[:,cols]

abs_wq_df = pd.merge(wq_df,HNSs_abs,on = 'Name',how = 'outer')

abs_wq_df.drop(columns = ['Lab Number','ID#'],inplace = True)

abs_wq_df.rename(columns = {'NH4N':'Ammonium-N','NO3N':'Nitrate-N','B':'Boron',
                            'Ca':'Calcium','Cu':'Copper','Fe':'Iron','K':'Potassium',
                            'Mg':'Magnesium','Mn':'Manganese','Mo':'Molybdenum',
                            'P':'Phosphorus','S':'Sulfate','Zn':'Zinc'},inplace = True)

abs_wq_df.drop(columns = 'TKN',inplace = True)
abs_wq_df.drop(columns = 'EC',inplace = True)

#%% convert 'S' to Sulfate

mw_sulfate = 96.06 # molar mass of sulfate

mw_S = 32.06 # molar mass of sulfur

abs_wq_df['Sulfate'] = abs_wq_df.Sulfate*mw_sulfate/mw_S

#%% Export dataframe

abs_wq_df.to_csv(inter_dir+'abs-wq_HNSs_df.csv',index=False)
