# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:57:51 2022

@author: jbarrett.carter
"""

#%% import libraries

import pandas as pd
# import numpy as np
import os
import datetime as dt
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
# from sklearn.metrics import mean_squared_error as MSE
import xgboost as xgb
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler

#for looking up available scorers
import sklearn.metrics
sorted(sklearn.metrics.SCORERS.keys())

from joblib import dump

#%% Set paths and bring in data

user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
# path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\water_quality-spectroscopy' #for laptop
path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')
figure_dir = r'C:\Users\\'+ user + r'\OneDrive\Research\PhD\Communications\Images\Stream results\XGB'

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

n_est = 500
e_stop = 3

XGBR = xgb.XGBRegressor(n_estimators = n_est,random_state=iteration,booster = 'gbtree',
                        tree_method = 'exact',early_stopping_rounds = e_stop)

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

X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, 
                                                    random_state=iteration,
                                                    test_size = 0.2)

param_grid = {'max_depth':stats.randint(2,50),
              'learning_rate':stats.uniform(scale=0.2)}

clf = RandomizedSearchCV(XGBR,
                         param_grid,n_iter = 20,
                         scoring = 'neg_mean_squared_error',
                         random_state = iteration)

train_start = dt.datetime.now()

clf.fit(X_train,y_train,eval_set = [(X_eval,y_eval)])

train_stop = dt.datetime.now()

train_stop_str = str(train_stop)
stop_date = train_stop_str.split(sep = ' ')[0]
stop_time = train_stop_str.split(sep = ' ')[1]
stop_time = str(float(stop_time.split(':')[0])+float(stop_time.split(':')[1])/60+float(stop_time.split(':')[2])/3600)
train_stop_str = stop_date+'_'+stop_time

train_time = train_stop - train_start

mod_opt = clf.best_estimator_
Y_hat = list(mod_opt.predict(X_test))
Y_hat_train = list(mod_opt.predict(X_train))

plt.figure()
plt.scatter(y_test,Y_hat)
plt.ylabel('Predicted')
plt.xlabel('True')
plt.title(s)
plt.savefig(os.path.join(figure_dir,f'XGB_{s}_11_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)

cv_results = pd.DataFrame(clf.cv_results_)
cv_results['learning_rate']=cv_results.params.apply(lambda x: x['learning_rate'])
cv_results['max_depth']=cv_results.params.apply(lambda x: x['max_depth'])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(cv_results.learning_rate,cv_results.max_depth,cv_results.mean_test_score)
ax.set_xlabel('learning rate')
ax.set_ylabel('max depth')
ax.set_zlabel('negative MSE')
plt.savefig(os.path.join(figure_dir,f'XGB_{s}_3dparams_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)
            

plt.figure()
plt.scatter(cv_results.learning_rate,cv_results.mean_test_score)
plt.xlabel('learning rate')
plt.ylabel('negative MSE')
plt.savefig(os.path.join(figure_dir,f'XGB_{s}_lr_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)

plt.figure()
plt.scatter(cv_results.max_depth,cv_results.mean_test_score)
plt.xlabel('max depth')
plt.ylabel('negative MSE')
plt.savefig(os.path.join(figure_dir,f'XGB_{s}_md_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)

filename = f'XGB_{s}_{train_stop_str}.joblib'
pickle_path = os.path.join(output_dir,'picklejar',filename)
dump(clf,pickle_path)