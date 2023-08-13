# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:09:45 2022

@author: jbarrett.carter
"""

#%% import libraries

from joblib import load
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from matplotlib.axis import Tick
# import scipy
# from scipy import stats
# import seaborn as sns

import datetime as dt
# import matplotlib.pyplot as plt
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
# import sklearn.metrics
# sorted(sklearn.metrics.SCORERS.keys())

from sklearn.metrics import r2_score

from joblib import dump

from sklearn.base import BaseEstimator

#%% make custom estimator combining PCA and XGB

class pca_xgb(BaseEstimator):
    
    def __init__(self, max_depth=None, learning_rate=None, 
                 n_components=None, detect_lim=None,
                 random_state=None):
        
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_components = n_components
        self.random_state = random_state
        self.detect_lim = detect_lim
        
    def fit(self, X, y):
        
        self.pca=PCA(n_components=self.n_components,
                     random_state=self.random_state)
        self.XGBR = xgb.XGBRegressor(n_estimators = 100,
                                     booster = 'gbtree',
                                     tree_method = 'exact',
                                     random_state=self.random_state,
                                     learning_rate=self.learning_rate,
                                     max_depth=self.max_depth)
        
        keep = y>self.detect_lim
        
        y = y[keep]
        
        X = X.loc[keep,:]
    
        self.pca_fitted = self.pca.fit(X)
        
        X = pd.DataFrame(self.pca.fit_transform(X))
        
        self.scaler_fitted = MinMaxScaler().fit(X)
        
        X = self.scaler_fitted.transform(X)
        
        self.XGBR_fitted=self.XGBR.fit(X,y)
        
        return self
    
    
    def predict(self, X):
        
        X = pd.DataFrame(self.pca_fitted.transform(X))
        X = self.scaler_fitted.transform(X)
        self.y_hat = pd.Series(self.XGBR_fitted.predict(X))
        self.y_hat[self.y_hat<self.detect_lim]=self.detect_lim
        
        return(self.y_hat)
    
    
    def set_params(self, **params):
        # if not params:
        #     return self
    
        # for key, value in params.items():
        #     if hasattr(self, key):
        #         setattr(self, key, value)
        #     else:
        #         self.kwargs[key] = value
        
        for param, value in params.items():
            setattr(self, param, value)
                
        return self
    
#%% Set paths and bring in models

user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
output_dir=os.path.join(path_to_wqs,'Hydroponics/outputs/')
inter_dir=os.path.join(path_to_wqs,'Hydroponics/intermediates/')

abs_wq_df_fn = 'abs-wq_HNSr_df.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)

filename = 'HNSr_XGB-PCA_Iron_It0.joblib'
pickle_path = os.path.join(output_dir,'picklejar',filename)

#bring in model
clf = load(pickle_path)

#%% Test model

input_df = abs_wq_df
s = 'Iron'
iteration = 0

Y = input_df[s]
keep = pd.notna(Y)
keep = (keep) & (Y>0)
X = input_df.loc[keep,'band_1':'band_1024']
Y = Y[keep]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=iteration,
                                                                test_size = 0.3)

mod = clf.best_estimator_

y_hat = mod.predict(X)

plt.figure()
plt.scatter(Y,y_hat)
plt.ylabel('Predicted')
plt.xlabel('True')

#%% Look at model tuning parameters

tuning_results = clf.cv_results_
scores = np.array(tuning_results['mean_test_score']*-1)
learning_rate = np.array(tuning_results['param_learning_rate'])
max_depth = np.array(tuning_results['param_max_depth'])

#%% Make plots

fig, axs = plt.subplots(1,2)
axs[0].plot(learning_rate,scores,'o')
axs[0].set_xlabel('learning_rate (eta)')
axs[0].set_ylabel('MAE (mg/L)')

axs[1].plot(max_depth,scores,'o')
axs[1].set_xlabel('maximum tree depth')
w = axs[1].get_yaxis() 
     
Tick.set_visible(w, False)

#plt.scatter(learning_rate,scores)

#%% test for significance

X_lm = pd.DataFrame({'learning_rate':learning_rate,'max_depth':max_depth})
X_lm['learning_rate']=X_lm['learning_rate'].apply(lambda x: float(x))
X_lm['max_depth']=X_lm['max_depth'].apply(lambda x: float(x))
X_lm = sm.add_constant(X_lm)
X_lm['axm']=X_lm.learning_rate*X_lm.max_depth

y = pd.DataFrame({'scores':scores})

lr_1 = sm.OLS(y, X_lm).fit()

print(lr_1.summary())

#%%
plt.figure()
plt.scatter(learning_rate*max_depth,scores)
plt.figure()
plt.scatter(learning_rate,max_depth)
