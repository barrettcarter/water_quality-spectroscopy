# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:57:51 2022

@author: jbarrett.carter
"""

#%% import libraries

import pandas as pd
import numpy as np
import os
# import datetime as dt
import matplotlib.pyplot as plt
# import scipy
from scipy import stats
# import seaborn as sns
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.utils import resample
from sklearn.metrics import mean_squared_error as MSE
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#for looking up available scorers
import sklearn.metrics
sorted(sklearn.metrics.SCORERS.keys())

from joblib import dump

#%% Set paths and bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')

abs_wq_df_fn = 'abs_wq_df_streams.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)

#%% seperate into filtered and unfiltered sample sets

abs_wq_df_fil = abs_wq_df.loc[abs_wq_df['Filtered']==True,:]
abs_wq_df_unf = abs_wq_df.loc[abs_wq_df['Filtered']==False,:]

#%% Make example xgb model
input_df = abs_wq_df
s = 'Nitrate-N'
iteration = 0

XGBR = xgb.XGBRegressor(n_estimators = 100,random_state=iteration,booster = 'gbtree',
                        tree_method = 'exact')

Y = input_df[s]
            
keep = pd.notna(Y)

# keep = pd.notna(Y) & (Y>0.15)

# if sum(keep)<10:
#     keep = pd.notna(Y)

X = input_df.loc[keep,'band_1':'band_1024']

# dimensional reduction
# pca = PCA(n_components = 20)
# X = pd.DataFrame(pca.fit_transform(X))
# X = MinMaxScaler(X)

Y = Y[keep]

X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    random_state=iteration,
                                                    test_size = 0.3)

param_grid = {'max_depth':stats.randint(2,10),
              'learning_rate':stats.uniform(scale=0.5)}

clf = RandomizedSearchCV(XGBR,
                         param_grid,n_iter = 10,
                         scoring = 'neg_mean_absolute_error',
                         random_state = iteration)

clf.fit(X_train,y_train)

mod_opt = clf.best_estimator_
Y_hat = list(mod_opt.predict(X_test))
Y_hat_train = list(mod_opt.predict(X_train))

plt.figure()
plt.scatter(y_test,Y_hat)
plt.ylabel('Predicted')
plt.xlabel('True')

filename = f'XGB_{s}_It{iteration}.joblib'
pickle_path = os.path.join(output_dir,'picklejar',filename)
dump(clf,pickle_path)