# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:36:58 2021

@author: barre
"""

import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import seaborn as sns
# IMPORTANT: Don't have the baseDir and saveDir be the same
user = os.getlogin() 
abs_df_dir='C:/Users/'+user+'/OneDrive/Documents/Data/Inputs/abs/'

abs_df_fn = 'abs_df_u2d.csv'

# Bring in data
abs_df=pd.read_csv(abs_df_dir+abs_df_fn)

hat = abs_df.loc[abs_df.Name=='hat',:]
hat = hat.reset_index(drop=True)

swb = abs_df.loc[abs_df.Name=='swb',:]
swb = swb.reset_index(drop=True)

hogdn = abs_df.loc[abs_df.Name=='hogdn',:]
hogdn = hogdn.reset_index(drop=True)

abs_small = abs_df.loc[(abs_df.Name=='hat') | (abs_df.Name=='swb'),:]
abs_small = abs_small.reset_index(drop=True)

bands = np.linspace(184.2, 667.6,num = 1024)
bands = np.around(bands,2)
cols = abs_small.columns.to_numpy()
cols[7:1031]=bands

abs_small.columns=cols

abs_long_name = pd.melt(abs_small,id_vars=['Name'],value_vars=list(bands),
                        var_name='wavelength',value_name='absorbance')

abs_long_filt = pd.melt(abs_small,id_vars=['Filtered'],value_vars=list(bands),
                        var_name='wavelength',value_name='absorbance')

abs_long_small = abs_long_name

abs_long_small['Filtered']=abs_long_filt['Filtered']

sns.set_theme(font_scale = 1.25,style='ticks')
abs_plots = sns.relplot(
    data=abs_long_small, kind="line",
    x="wavelength", y="absorbance", col="Name",
    hue="Filtered", style="Filtered",
)

abs_plots.set_axis_labels(x_var='Wavelength (nm)',y_var='Absorbance')

sns.set_theme(font_scale = 1.25)
ex_plot = sns.relplot(data=abs_long_small)
ex_plot.set_axis_labels(x_var='wavelength (nm)')