# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:09:45 2021

@author: jbarrett.carter
"""

import pandas as pd
import numpy as np
import os
# import datetime as dt
import matplotlib.pyplot as plt
# import scipy
# from scipy import stats
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor

#for looking up available scorers
import sklearn.metrics
sorted(sklearn.metrics.SCORERS.keys())

#%%

### Set paths and bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')

abs_wq_df_fn = 'abs_wq_df_streams.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)