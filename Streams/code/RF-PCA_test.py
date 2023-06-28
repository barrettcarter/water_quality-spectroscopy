# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:57:51 2022

@author: jbarrett.carter
"""

#%% import libraries

import pandas as pd
import numpy as np
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
# import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.preprocessing import MinMaxScaler

#for looking up available scorers
import sklearn.metrics
sorted(sklearn.metrics.SCORERS.keys())

from joblib import dump

from sklearn.base import BaseEstimator

#%% Set paths and bring in data

user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
# path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\water_quality-spectroscopy' #for laptop
path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')
figure_dir = r'C:\Users\\'+ user + r'\OneDrive\Research\PhD\Communications\Images\Stream results\RF'

abs_wq_df_fn = 'abs_wq_df_streams.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)

#%% seperate into filtered and unfiltered sample sets

abs_wq_df_fil = abs_wq_df.loc[abs_wq_df['Filtered']==True,:]
abs_wq_df_unf = abs_wq_df.loc[abs_wq_df['Filtered']==False,:]

#%% Make example model

input_df = abs_wq_df
s = 'Phosphate-P'
iteration = 0

n_est = 100
e_stop = 3
detect_lim = 0

reg = RF(n_estimators = 100,random_state=iteration)

Y = input_df[s]

keep = pd.notna(Y)

X = input_df.loc[keep,'band_1':'band_1024']

Y = Y[keep]

X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    random_state=iteration,
                                                    test_size = 0.3)
            
# keep = pd.notna(Y) & (Y>detect_lim)

# if sum(keep)<10:
#     keep = pd.notna(Y)

keep_tr = y_train > detect_lim
keep_te = y_test > detect_lim

y_train = y_train[keep_tr]
y_test = y_test[keep_te]

X_train = X_train.loc[keep_tr,:]
X_test = X_test.loc[keep_te,:]

# dimensional reduction
n_comp = 20

for n_comp in [10]:
    
    # X = input_df.loc[keep,'band_1':'band_1024']

    pca = PCA(n_components = n_comp,random_state = iteration)
    # X = pd.DataFrame(pca.fit_transform(X))
    # X = MinMaxScaler().fit_transform(X)
    
    # Y = Y[keep]
    
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, 
    #                                                     random_state=iteration,
    #                                                     test_size = 0.3)
    
    pca_train = pca.fit(X_train)
    
    X_train = pd.DataFrame(pca_train.transform(X_train))
    
    scaler_train = MinMaxScaler().fit(X_train)
    
    X_train = scaler_train.transform(X_train)
    
    
    X_test = pd.DataFrame(pca_train.transform(X_test))
    X_test = scaler_train.transform(X_test)
    
    param_grid = {'max_features':stats.uniform(loc = 5/1024,scale = 200/1024),
                          'ccp_alpha':stats.uniform(scale=0.0001)}
    
    clf = RandomizedSearchCV(reg,
                             param_grid,n_iter = 20,
                             scoring = 'neg_mean_squared_error',
                             random_state = iteration)
    
    train_start = dt.datetime.now()
    
    clf.fit(X_train,y_train)
    
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
    
    ccp_alpha = mod_opt.ccp_alpha
    max_features = mod_opt.max_features
    n_est = mod_opt.n_estimators
    
    res_tr = Y_hat_train - y_train
    se_tr = res_tr**2
    sse_tr = sum(se_tr)
    mse_tr = np.mean(se_tr)
    rmse_tr = np.sqrt(mse_tr)
    
    res_te = Y_hat - y_test
    se_te = res_te**2
    sse_te = sum(se_te)
    mse_te = np.mean(se_te)
    rmse_te = np.sqrt(mse_te)
    
    min11 = min([min(y_test),min(y_train),min(Y_hat),min(Y_hat_train)])
    max11 = max([max(y_test),max(y_train), max(Y_hat),max(Y_hat_train)])
    
    y_text = min11+(max11-min11)*0
    x_text = max11+(max11-min11)*0.05
    
    plt.figure()
    plt.scatter(y_train,Y_hat_train)
    plt.scatter(y_test,Y_hat)
    plt.plot([min11,max11],[min11,max11],'--k')
    plt.ylabel('Predicted')
    plt.xlabel('True')
    plt.title(s)
    plt.text(x_text,y_text,'$RMSE_{tr} =$'+str(np.round(rmse_tr,2))+'\n'
                    +'$RMSE_{te} =$'+str(np.round(rmse_te,2))+'\n'
                    +'$alpha =$'+'{:.2e}'.format(ccp_alpha)+'\n'
                    +'$MF =$'+str(int(max_features*1024))+'\n'
                    +'$n_{est} =$'+str(int(n_est))+'\n'
                    +'$n_{comp} =$'+str(int(n_comp))+'\n'
                    +'$det lim =$'+str(np.round(detect_lim,2)), fontsize = 12)
    
    plt.savefig(os.path.join(figure_dir,f'RF_{s}_11_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)
    
    cv_results = pd.DataFrame(clf.cv_results_)
    cv_results['ccp_alpha']=cv_results.params.apply(lambda x: x['ccp_alpha'])
    cv_results['max_features']=cv_results.params.apply(lambda x: x['max_features'])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(cv_results.ccp_alpha,cv_results.max_features,cv_results.mean_test_score)
    ax.set_xlabel('learning rate')
    ax.set_ylabel('max depth')
    ax.set_zlabel('negative MSE')
    
    plt.savefig(os.path.join(figure_dir,f'RF_{s}_3dparams_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)
                
    
    plt.figure()
    plt.scatter(cv_results.ccp_alpha,cv_results.mean_test_score)
    plt.xlabel('learning rate')
    plt.ylabel('negative MSE')
    
    plt.savefig(os.path.join(figure_dir,f'RF_{s}_lr_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)
    
    plt.figure()
    plt.scatter(cv_results.max_features,cv_results.mean_test_score)
    plt.xlabel('max depth')
    plt.ylabel('negative MSE')
    
    plt.savefig(os.path.join(figure_dir,f'RF_{s}_md_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)
    
    # filename = f'RF_{s}_{train_stop_str}.joblib'
    # pickle_path = os.path.join(output_dir,'picklejar',filename)
    # dump(clf,pickle_path)
    
#%% make custom estimator combining PCA and RF

class pca_RF(BaseEstimator):
    
    def __init__(self, max_features=None, ccp_alpha=None, 
                 n_components=None, detect_lim=None,
                 random_state=None):
        
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.n_components = n_components
        self.random_state = random_state
        self.detect_lim = detect_lim
        
    def fit(self, X, y):
        
        self.pca=PCA(n_components=self.n_components,
                     random_state=self.random_state)
        
        self.reg = RF(n_estimators = 100,
                      random_state=self.random_state,
                      ccp_alpha=self.ccp_alpha,
                      max_features=self.max_features)
        
        keep = y>self.detect_lim
        
        y = y[keep]
        
        X = X.loc[keep,:]
    
        self.pca_fitted = self.pca.fit(X)
        
        X = pd.DataFrame(self.pca.fit_transform(X))
        
        self.scaler_fitted = MinMaxScaler().fit(X)
        
        X = self.scaler_fitted.transform(X)
        
        self.reg_fitted=self.reg.fit(X,y)
        
        return self
    
    
    def predict(self, X):
        
        X = pd.DataFrame(self.pca_fitted.transform(X))
        X = self.scaler_fitted.transform(X)
        self.y_hat = pd.Series(self.reg_fitted.predict(X))
        self.y_hat[self.y_hat<self.detect_lim]=np.nan
        
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


#%% test custom estimator

input_df = abs_wq_df
s = 'Phosphate-P'
iteration = 0

Y = input_df[s]
            
keep = pd.notna(Y)

# keep = pd.notna(Y) & (Y>0.2)

# if sum(keep)<10:
#     keep = pd.notna(Y)
    
Y = Y[keep]
    
X = input_df.loc[keep,'band_1':'band_1024']

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    random_state=iteration,
                                                    test_size = 0.3)

# max_features = 179/1024
# ccp_alpha = 0.0000778
# n_comp = 10
# n_est = 100
# detect_lim = 0.2

## create model object using parameters from previous example model
## the results of this model object should be identical to the previous model

mod = pca_RF(max_features = max_features, ccp_alpha = ccp_alpha,
             n_components = n_comp,random_state=iteration,
             detect_lim = detect_lim)

mod_fitted = mod.fit(X=X_train,y=y_train)

Y_hat = mod_fitted.predict(X_test)
Y_hat_train = mod_fitted.predict(X_train)

res_tr = Y_hat_train - y_train
se_tr = res_tr**2
sse_tr = sum(se_tr)
mse_tr = np.mean(se_tr)
rmse_tr = np.sqrt(mse_tr)

res_te = Y_hat - y_test
se_te = res_te**2
sse_te = sum(se_te)
mse_te = np.mean(se_te)
rmse_te = np.sqrt(mse_te)

min11 = min([min(y_test),min(y_train),min(Y_hat),min(Y_hat_train)])
max11 = max([max(y_test),max(y_train), max(Y_hat),max(Y_hat_train)])

y_text = min11+(max11-min11)*0
x_text = max11+(max11-min11)*0.05

plt.figure()
plt.scatter(y_train,Y_hat_train)
plt.scatter(y_test,Y_hat)
plt.plot([min11,max11],[min11,max11],'--k')
plt.ylabel('Predicted')
plt.xlabel('True')
plt.title(s)
plt.text(x_text,y_text,'$RMSE_{tr} =$'+str(np.round(rmse_tr,2))+'\n'
                +'$RMSE_{te} =$'+str(np.round(rmse_te,2))+'\n'
                +'$alpha =$'+'{:.2e}'.format(ccp_alpha)+'\n'
                +'$MF =$'+str(int(max_features*1024))+'\n'
                +'$n_{est} =$'+str(int(n_est))+'\n'
                +'$n_{comp} =$'+str(int(n_comp))+'\n'
                +'$det lim =$'+str(np.round(detect_lim,2)), fontsize = 12)

#%% try changing parameters

mod_fitted.set_params(max_features=4)

y_hat = mod_fitted.predict(X_test)
y_hat_train = mod_fitted.predict(X_train)

plt.scatter(y_test,y_hat)
plt.xlabel('True')
plt.ylabel('Predicted')

# seems to work

#%% calibrate combined model

input_df = abs_wq_df
s = 'OP'
iteration = 0

Y = input_df[s]
            
keep = pd.notna(Y)

# keep = pd.notna(Y) & (Y>0.2)

# if sum(keep)<10:
#     keep = pd.notna(Y)
    
X = input_df.loc[keep,'band_1':'band_1024']

Y = Y[keep]

X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    random_state=iteration,
                                                    test_size = 0.3)

# X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, 
#                                                     random_state=iteration,
#                                                     test_size = 0.2)

mod = pca_RF(random_state=iteration)

param_grid = {'max_features':stats.randint(1,4),
              'ccp_alpha':stats.uniform(loc=0.02,scale=0.3),
              'n_components':stats.randint(10,50)
              'detect_lim':stats.uniform(scale=0.5*max(y_train))}

clf = RandomizedSearchCV(mod,
                         param_grid,n_iter = 100,
                         scoring = 'neg_mean_squared_error',
                         random_state = iteration)

train_start = dt.datetime.now()

clf.fit(X_train,y_train)

train_stop = dt.datetime.now()

#%% post-processing

train_stop_str = str(train_stop)
stop_date = train_stop_str.split(sep = ' ')[0]
stop_time = train_stop_str.split(sep = ' ')[1]
stop_time = str(float(stop_time.split(':')[0])+float(stop_time.split(':')[1])/60+float(stop_time.split(':')[2])/3600)
train_stop_str = stop_date+'_'+stop_time

train_time = train_stop - train_start

mod_opt = clf.best_estimator_
Y_hat = list(mod_opt.predict(X_test))
Y_hat_train = list(mod_opt.predict(X_train))

ccp_alpha = mod_opt.ccp_alpha
max_features = mod_opt.max_features
n_comp=mod_opt.n_components
# n_est = mod_opt.n_estimators
detect_lim = mod_opt.detect_lim

res_tr = Y_hat_train - y_train
se_tr = res_tr**2
sse_tr = sum(se_tr)
mse_tr = np.mean(se_tr)
rmse_tr = np.sqrt(mse_tr)

res_te = Y_hat - y_test
se_te = res_te**2
sse_te = sum(se_te)
mse_te = np.mean(se_te)
rmse_te = np.sqrt(mse_te)

min11 = min([min(y_test),min(Y_hat),min(Y_hat_train),min(y_train)])
max11 = max([max(y_test),max(Y_hat),max(Y_hat_train),max(y_train)])

y_text = min11+(max11-min11)*0
x_text = max11+(max11-min11)*0.1

plt.figure()
plt.scatter(y_train,Y_hat_train)
plt.scatter(y_test,Y_hat)
plt.plot([min11,max11],[min11,max11],'--k')
plt.ylabel('Predicted')
plt.xlabel('True')
plt.title(s)
plt.text(x_text,y_text,'$RMSE_{tr} =$'+str(np.round(rmse_tr,2))+'\n'
                +'$RMSE_{te} =$'+str(np.round(rmse_te,2))+'\n'
                +'$alpha =$'+'{:.2e}'.format(ccp_alpha)+'\n'
                +'$MF =$'+str(int(max_features))+'\n'
                +'$n_{comp} =$'+str(int(n_comp))+'\n'
                +'$detect lim =$'+str(np.round(detect_lim,2)), fontsize = 12)

plt.savefig(os.path.join(figure_dir,f'RF_{s}_11_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)

cv_results = pd.DataFrame(clf.cv_results_)
cv_results['ccp_alpha']=cv_results.params.apply(lambda x: x['ccp_alpha'])
cv_results['max_features']=cv_results.params.apply(lambda x: x['max_features'])
cv_results['n_components']=cv_results.params.apply(lambda x: x['n_components'])
cv_results['detect_lim']=cv_results.params.apply(lambda x: x['detect_lim'])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(cv_results.ccp_alpha,cv_results.max_features,cv_results.mean_test_score)
ax.set_xlabel('learning rate')
ax.set_ylabel('max depth')
ax.set_zlabel('negative MSE')

plt.savefig(os.path.join(figure_dir,f'RF_{s}_3dparams_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)
            

plt.figure()
plt.scatter(cv_results.ccp_alpha,cv_results.mean_test_score)
plt.xlabel('learning rate')
plt.ylabel('negative MSE')

plt.savefig(os.path.join(figure_dir,f'RF_{s}_lr_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)

plt.figure()
plt.scatter(cv_results.max_features,cv_results.mean_test_score)
plt.xlabel('max depth')
plt.ylabel('negative MSE')

plt.savefig(os.path.join(figure_dir,f'RF_{s}_md_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)


plt.figure()
plt.scatter(cv_results.n_components,cv_results.mean_test_score)
plt.xlabel('n components')
plt.ylabel('negative MSE')

plt.savefig(os.path.join(figure_dir,f'RF_{s}_nc_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)

plt.figure()
plt.scatter(cv_results.detect_lim,cv_results.mean_test_score)
plt.xlabel('detection limit')
plt.ylabel('negative MSE')

plt.savefig(os.path.join(figure_dir,f'RF_{s}_dl_{train_stop_str}.png'),bbox_inches = 'tight',dpi = 300)

filename = f'RF_{s}_{train_stop_str}.joblib'
pickle_path = os.path.join(output_dir,'picklejar',filename)
dump(clf,pickle_path)

#%% scratch

X_trans = pd.DataFrame(mod_fitted.pca_fitted.transform(X_test))
