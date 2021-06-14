# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:39:55 2021

@author: jbarrett.carter
"""

import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor

# for looking up available scorers
# import sklearn.metrics
# sorted(sklearn.metrics.SCORERS.keys())

# IMPORTANT: Don't have the baseDir and saveDir be the same
user = os.getlogin() 
abs_df_dir='C:/Users/'+user+'/OneDrive/Documents/Data/Inputs/abs/'
wq_df_dir='C:/Users/'+user+'/OneDrive/Documents/Data/Inputs/wq/'


wq_df_fn = 'wq_kalera_df.csv'
abs_df_fn = 'abs_df_u2d.csv'

# Bring in data
abs_df=pd.read_csv(abs_df_dir+abs_df_fn)
wq_df=pd.read_csv(wq_df_dir+wq_df_fn)

# Select only kalera samples

abs_df = abs_df.loc[(abs_df.Name == 'kalera1') | (abs_df.Name == 'kalera2')\
                    | (abs_df.Name == 'kalera3'),:]
abs_df = abs_df.reset_index(drop = True)
abs_df.loc[7,'Name']='kalera1' # this sample was labelled incorrectly (probably)
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
wq_cols = list(wq_df.columns[0:4])
wq_cols.append('Name')
wq_df.columns=wq_cols
wq_df['ID']=wq_df.Name+wq_df.Date_col

# concs_ind = range(int((wq_df.shape[0]-1)/3))
# concs_df = wq_df.pivot(index = range(268),columns = 'Species',values = 'Conc')

species = wq_df.Species.unique()
species = np.append(species,'ID')
aw_df_cols = np.append(species,abs_df.columns)
abs_wq_df = abs_df

abs_wq_df.loc[:,species] = -0.1
abs_wq_df['ID']=abs_wq_df.Name+abs_wq_df.Date_col

species = np.delete(species,len(species)-1)

abs_wq_df = abs_wq_df[aw_df_cols]

for wq_row in range(wq_df.shape[0]):
    for abs_row in range(abs_wq_df.shape[0]):
        if wq_df.ID[wq_row]==abs_wq_df.ID[abs_row]:
            for s in species:
                if wq_df.Species[wq_row] == s:
                    abs_wq_df.loc[abs_row,s]=wq_df.Value[wq_row]
                             

#################################################################


### Tuning the models

## Nitrate
## PlSR

# Best r_sq achieved by not including name or filtered with abs

param_grid = [{'n_components':np.arange(1,5)}]

# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
X = abs_wq_df.loc[:,'band_1':'band_1024']
Y = abs_wq_df['Nitrate-N']
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
# X = abs_wq_df.loc[keep,:]
# Y = abs_wq_df.Nitrate[keep].to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
# X = pd.concat([name_dum,X],axis=1).to_numpy()
# X = name_dum


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2,
                                                    test_size = 0.2)
# test_names = X_test.Name.reset_index(drop=True)
# test_filt = X_test.Filtered.reset_index(drop=True)

# X_train = X_train.loc[:,'band_1':'band_1024']
# X_test = X_test.loc[:,'band_1':'band_1024']
pls = PLSRegression()
clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_absolute_error')
# clf.cv_results_
# clf = GridSearchCV(pls,param_grid)
clf.fit(X_train,y_train)
n_comp = clf.best_params_['n_components']
pls_opt = clf.best_estimator_
Y_hat = pls_opt.predict(X_test)
Y_hat_train = pls_opt.predict(X_train)

r_sq = pls_opt.score(X_test,y_test)
r_sq_train = pls_opt.score(X_train,y_train)

# plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(np.concatenate((y_test,Y_hat[:,0]))),
                     max(np.concatenate((y_test,Y_hat[:,0]))))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate (mg/L)')
plt.ylabel('Predicted Nitrate (mg/L)')
plt.text(170,200,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.legend()
plt.show()

line11 = np.linspace(min(np.concatenate((y_train,Y_hat_train[:,0]))),
                     max(np.concatenate((y_train,Y_hat_train[:,0]))))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_train,Y_hat_train,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate (mg/L)')
plt.ylabel('Predicted Nitrate (mg/L)')
plt.text(0.8*max(line11),min(line11),r'$r^2 =$'+str(np.round(r_sq_train,3)))
plt.legend()
plt.show()

# make better plot
data_out = pd.DataFrame({'y_test':y_test,'y_pred':Y_hat[:,0]})
data_out = pd.concat([data_out,test_names,test_filt],axis=1)

sns.set_theme(style ='ticks',font_scale = 1.25,
              palette = 'colorblind')

g = sns.relplot(
    data=data_out,
    x = 'y_test',
    y = 'y_pred',
    hue = 'Filtered',
    style = 'Name',
    s = 60
    )

plt.plot(line11,line11,label= '1:1 line',color = 'k',ls = 'dashed')
plt.xlabel('Lab Measured Nitrate (mg/L)')
plt.ylabel('Predicted Nitrate (mg/L)')
plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))

for s in species:
    Y = abs_wq_df[s]
    # name_dum = pd.get_dummies(abs_wq_df['Name'])
    # filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
    # X = abs_wq_df.loc[keep,:]
    # Y = abs_wq_df.Nitrate[keep].to_numpy()
    # name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
    # X = pd.concat([name_dum,X],axis=1).to_numpy()
    # X = name_dum
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2,
                                                        test_size = 0.2)
    # test_names = X_test.Name.reset_index(drop=True)
    # test_filt = X_test.Filtered.reset_index(drop=True)
    
    # X_train = X_train.loc[:,'band_1':'band_1024']
    # X_test = X_test.loc[:,'band_1':'band_1024']
    pls = PLSRegression()
    clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_absolute_error')
    # clf.cv_results_
    # clf = GridSearchCV(pls,param_grid)
    clf.fit(X_train,y_train)
    n_comp = clf.best_params_['n_components']
    pls_opt = clf.best_estimator_
    Y_hat = pls_opt.predict(X_test)
    Y_hat_train = pls_opt.predict(X_train)
    
    r_sq = pls_opt.score(X_test,y_test)
    r_sq_train = pls_opt.score(X_train,y_train)
    
    # plt.plot(Y_hat,y_test,'b.')
    
    line11 = np.linspace(min(np.concatenate((y_test,Y_hat[:,0]))),
                         max(np.concatenate((y_test,Y_hat[:,0]))))
    
    # lr = LinearRegression().fit(Y_hat,y_test)
    # linelr = lr.predict(line11.reshape(-1,1))
    
    plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
    plt.plot(line11,line11,label= '1:1 line')
    # plt.plot(line11,linelr,label = 'regression line')
    plt.xlabel('Lab Measured '+s+' (mg/L)')
    plt.ylabel('Predicted '+s+' (mg/L)')
    plt.text(0.8*max(line11),min(line11),r'$r^2 =$'+str(np.round(r_sq,3)))
    plt.legend()
    plt.show()
    
    line11 = np.linspace(min(np.concatenate((y_train,Y_hat_train[:,0]))),
                         max(np.concatenate((y_train,Y_hat_train[:,0]))))
    
    # lr = LinearRegression().fit(Y_hat,y_test)
    # linelr = lr.predict(line11.reshape(-1,1))
    
    plt.plot(y_train,Y_hat_train,'o',markersize = 4, label = 'predictions')
    plt.plot(line11,line11,label= '1:1 line')
    # plt.plot(line11,linelr,label = 'regression line')
    plt.xlabel('Lab Measured '+s+' (mg/L)')
    plt.ylabel('Predicted '+s+' (mg/L)')
    plt.text(0.8*max(line11),min(line11),r'$r^2 =$'+str(np.round(r_sq_train,3)))
    plt.legend()
    plt.show()

## Random Forest (nitrate)

# Best r_sq achieved with...
# max_features ~70 seems to do best, but this varies a lot

param_grid = [{'max_features':np.arange(10,110,10)}]
# param_grid = [{'max_features':np.arange(60,80,2)}]

# keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
X = abs_wq_df.loc[:,'band_1':'band_1024']
Y = abs_wq_df['Nitrate-N']
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
# X = abs_wq_df.loc[keep,'band_1':'band_1024']
# Y = abs_wq_df.Nitrate[keep].to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
# X = pd.concat([name_dum,X],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
# test_names = X_test.Name.reset_index(drop=True)
# test_filt = X_test.Filtered.reset_index(drop=True)

# absorbance values
num_wavelengths = 2**10
step = 1024/num_wavelengths
wavelength_inds = np.arange(0,1024,step)
X_train = X_train.iloc[:,wavelength_inds]
X_test = X_test.iloc[:,wavelength_inds]

# abs and names
# X_train = pd.concat([name_dum.loc[X_train.index,:],
#                       X_train],axis=1)
# X_test = pd.concat([name_dum.loc[X_test.index,:],X_test],axis=1)

# abs and filtered
# X_train = pd.concat([filtered_dum.loc[X_train.index,1],
#                       X_train],axis=1)
# X_test = pd.concat([filtered_dum.loc[X_test.index,1],X_test],axis=1)

# abs, names, and filtered
# X_train = pd.concat([name_dum.loc[X_train.index,:],
#                       filtered_dum.loc[X_train.index,:],X_train],axis=1)
# X_test = pd.concat([name_dum.loc[X_test.index,:],
#                     filtered_dum.loc[X_test.index,:],X_test],axis=1)

rf = RandomForestRegressor()
clf = GridSearchCV(rf,param_grid,scoring = 'neg_mean_squared_error')
clf.fit(X_train,y_train)
max_feat = clf.best_params_['max_features']
rf_opt = clf.best_estimator_
r_sq = rf_opt.score(X_test,y_test)

# to iterate

max_feats = np.zeros(10)
r_squares = np.zeros(10)

for i in range(10):
    rf = RandomForestRegressor()
    clf = GridSearchCV(rf,param_grid)
    clf.fit(X_train,y_train)
    max_feat = clf.best_params_['max_features']
    rf_opt = clf.best_estimator_
    r_sq = rf_opt.score(X_test,y_test)
    
    max_feats[i]=max_feat
    r_squares[i]=r_sq

Y_hat = rf_opt.predict(X_test)
y_hat_train = rf_opt.predict(X_train)

# plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(np.concatenate((y_test,Y_hat))),
                     max(np.concatenate((y_test,Y_hat))))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate (mg/L)')
plt.ylabel('Predicted Nitrate (mg/L)')
plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.legend()
plt.show()

plt.plot(y_train,y_hat_train,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate (mg/L)')
plt.ylabel('Predicted Nitrate (mg/L)')
# plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.legend()
plt.show()

# make better plot
data_out = pd.DataFrame({'y_test':y_test,'y_pred':Y_hat[:,0]})
data_out = pd.concat([data_out,test_names,test_filt],axis=1)

sns.set_theme(style ='ticks',font_scale = 1.25,
              palette = 'colorblind')

g = sns.relplot(
    data=data_out,
    x = 'y_test',
    y = 'y_pred',
    hue = 'Filtered',
    style = 'Name',
    s = 60
    )

plt.plot(line11,line11,label= '1:1 line',color = 'k',ls = 'dashed')
plt.xlabel('Lab Measured Nitrate (mg/L)')
plt.ylabel('Predicted Nitrate (mg/L)')
plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))

## Phosphate

param_grid = [{'n_components':np.arange(1,5)}]

# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
X = abs_wq_df.loc[:,'band_1':'band_1024']
Y = abs_wq_df['Phosphorus']
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
# X = abs_wq_df.loc[keep,:]
# Y = abs_wq_df.Nitrate[keep].to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
# X = pd.concat([name_dum,X],axis=1).to_numpy()
# X = name_dum


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
# test_names = X_test.Name.reset_index(drop=True)
# test_filt = X_test.Filtered.reset_index(drop=True)

# X_train = X_train.loc[:,'band_1':'band_1024']
# X_test = X_test.loc[:,'band_1':'band_1024']
pls = PLSRegression()
clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_squared_error',cv=y_train.shape[0])
# clf = GridSearchCV(pls,param_grid)
clf.fit(X_train,y_train)
n_comp = clf.best_params_['n_components']
pls_opt = clf.best_estimator_
Y_hat = pls_opt.predict(X_test)
Y_hat_train = pls_opt.predict(X_train)

r_sq = pls_opt.score(X_test,y_test)
r_sq_train = pls_opt.score(X_train,y_train)

# plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(np.concatenate((y_test,Y_hat[:,0]))),
                     max(np.concatenate((y_test,Y_hat[:,0]))))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Phosphorus (mg/L)')
plt.ylabel('Predicted Phosphorus (mg/L)')
plt.text(32,24,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.legend()
plt.show()

line11 = np.linspace(min(np.concatenate((y_train,Y_hat_train[:,0]))),
                     max(np.concatenate((y_train,Y_hat_train[:,0]))))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_train,Y_hat_train,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Phosphorus (mg/L)')
plt.ylabel('Predicted Phosphorus (mg/L)')
plt.text(32,24,r'$r^2 =$'+str(np.round(r_sq_train,3)))
plt.legend()
plt.show()

# make better plot
data_out = pd.DataFrame({'y_test':y_test,'y_pred':Y_hat[:,0]})
data_out = pd.concat([data_out,test_names,test_filt],axis=1)

sns.set_theme(style ='ticks',font_scale = 1.25,palette = 'colorblind')

g = sns.relplot(
    data=data_out,
    x = 'y_test',
    y = 'y_pred',
    hue = 'Filtered',
    style = 'Name',
    s = 60
    )

plt.plot(line11,line11,label= '1:1 line',color = 'k',ls = 'dashed')
plt.xlabel('Lab Measured Phosphate (mg/L)')
plt.ylabel('Predicted Phosphate (mg/L)')
plt.text(0.5,0.2,r'$r^2 =$'+str(np.round(r_sq,3)))

coefs = pls.coef_

plt.plot(coefs[0:200])

#### Potassium

param_grid = [{'n_components':np.arange(1,5)}]

# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
X = abs_wq_df.loc[:,'band_1':'band_1024']
Y = abs_wq_df['Potassium']
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
# X = abs_wq_df.loc[keep,:]
# Y = abs_wq_df.Nitrate[keep].to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
# X = pd.concat([name_dum,X],axis=1).to_numpy()
# X = name_dum


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
# test_names = X_test.Name.reset_index(drop=True)
# test_filt = X_test.Filtered.reset_index(drop=True)

# X_train = X_train.loc[:,'band_1':'band_1024']
# X_test = X_test.loc[:,'band_1':'band_1024']
pls = PLSRegression()
clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_squared_error',cv=y_train.shape[0])
# clf = GridSearchCV(pls,param_grid)
clf.fit(X_train,y_train)
n_comp = clf.best_params_['n_components']
pls_opt = clf.best_estimator_
Y_hat = pls_opt.predict(X_test)
Y_hat_train = pls_opt.predict(X_train)

r_sq = pls_opt.score(X_test,y_test)
r_sq_train = pls_opt.score(X_train,y_train)

# plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(np.concatenate((y_test,Y_hat[:,0]))),
                     max(np.concatenate((y_test,Y_hat[:,0]))))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Potassium (mg/L)')
plt.ylabel('Predicted Potassium (mg/L)')
# plt.text(32,24,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.legend()
plt.show()

line11 = np.linspace(min(np.concatenate((y_train,Y_hat_train[:,0]))),
                     max(np.concatenate((y_train,Y_hat_train[:,0]))))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_train,Y_hat_train,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Potassium (mg/L)')
plt.ylabel('Predicted Potassium (mg/L)')
# plt.text(32,24,r'$r^2 =$'+str(np.round(r_sq_train,3)))
plt.legend()
plt.show()

## Random forest (phosphate)######################################
# good results obtained when abs and names are included
# dimensions/run time can be reduced by sampling from abs bands
# at regular increments, but max_features has to be around the same
# as number of bands included.

param_grid = [{'max_features':np.arange(10,1020,100)}]
# param_grid = [{'max_features':np.arange(60,80,2)}]

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Phosphate.to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
X = abs_wq_df.loc[keep,'band_1':'band_1024']
Y = abs_wq_df.Phosphate[keep].to_numpy()
name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
filtered_dum = pd.get_dummies(abs_wq_df['Filtered'][keep])
# X = pd.concat([name_dum,X],axis=1)
# X = name_dum

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
# test_names = X_test.Name.reset_index(drop=True)
# test_filt = X_test.Filtered.reset_index(drop=True)

# absorbance values
# num_wavelengths = 2**10
# step = 1024/num_wavelengths
# wavelength_inds = np.arange(0,1024,step)
# X_train = X_train.iloc[:,wavelength_inds]
# X_test = X_test.iloc[:,wavelength_inds]

# abs and names
X_train = pd.concat([name_dum.loc[X_train.index,:],
                      X_train],axis=1)
X_test = pd.concat([name_dum.loc[X_test.index,:],X_test],axis=1)

# abs and filtered
# X_train = pd.concat([filtered_dum.loc[X_train.index,1],
#                       X_train],axis=1)
# X_test = pd.concat([filtered_dum.loc[X_test.index,1],X_test],axis=1)

# abs, names, and filtered
# X_train = pd.concat([name_dum.loc[X_train.index,:],
#                       filtered_dum.loc[X_train.index,:],X_train],axis=1)
# X_test = pd.concat([name_dum.loc[X_test.index,:],
#                     filtered_dum.loc[X_test.index,:],X_test],axis=1)

rf = RandomForestRegressor()
clf = GridSearchCV(rf,param_grid)
clf.fit(X_train,y_train)
max_feat = clf.best_params_['max_features']
rf_opt = clf.best_estimator_
r_sq = rf_opt.score(X_test,y_test)

# to iterate

max_feats = np.zeros(10)
r_squares = np.zeros(10)

for i in range(10):
    rf = RandomForestRegressor()
    clf = GridSearchCV(rf,param_grid)
    clf.fit(X_train,y_train)
    max_feat = clf.best_params_['max_features']
    rf_opt = clf.best_estimator_
    r_sq = rf_opt.score(X_test,y_test)
    
    max_feats[i]=max_feat
    r_squares[i]=r_sq
    
Y_hat = rf_opt.predict(X_test)

rmse = MSE(y_test,Y_hat)

plt.plot(y_test,Y_hat)


### Using bootstrapping to obtain better performance metrics

## Nitrate (trained withough Kalera data), PLS

# Best r_sq achieved by including names and abs

param_grid = [{'n_components':np.arange(1,21)}]

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Nitrate
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])

# just absorbance values
X = abs_wq_df.loc[keep,'band_1':'band_1024']

# dummy variables
name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
filtered_dum = pd.get_dummies(abs_wq_df['Filtered'][keep])

# include name_dum
X = pd.concat([name_dum,X],axis=1)

# include filtered_dum
# X = pd.concat([filtered_dum,X],axis=1)

# include name_dum and filtered_dum
# X = pd.concat([filtered_dum,name_dum,X],axis=1)

Y = abs_wq_df.Nitrate[keep]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

num_b = 10
train_inds = np.zeros([X_train.shape[0],num_b])
test_inds = np.zeros([X_test.shape[0],num_b])
r_squares = np.zeros(num_b)
rmses = np.zeros(num_b)
errors = np.zeros([X_test.shape[0],num_b])
n_comps = np.zeros(num_b)
y_hats = np.zeros([X_test.shape[0],num_b])
y_trues = np.zeros([X_test.shape[0],num_b])

for b in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=b)
    X_test = resample(X_test,random_state = b)
    X_train = resample(X_train,random_state = b)
    y_train = resample(y_train,random_state = b)
    y_test = resample(y_test,random_state = b)
    
    train_inds[:,b]=X_train.index
    test_inds[:,b]=X_test.index

    # X_train = X_train.loc[:,'band_1':'band_1024']
    # X_test = X_test.loc[:,'band_1':'band_1024']
    pls = PLSRegression()
    clf = GridSearchCV(pls,param_grid)
    clf.fit(X_train,y_train)
    pls_opt = clf.best_estimator_
    Y_hat = pls_opt.predict(X_test)
    
    n_comp = clf.best_params_['n_components']
    r_sq = pls_opt.score(X_test,y_test)
    rmse = MSE(y_test,Y_hat)
    
    n_comps[b]=n_comp
    r_squares[b]=r_sq
    rmses[b]= rmse
    errors[:,b]=y_test-Y_hat[:,0]
    y_hats[:,b]=Y_hat[:,0]
    y_trues[:,b]=y_test

# plt.plot(Y_hat,y_test,'b.')

y_hats_flat = y_hats.flatten()
y_trues_flat = y_trues.flatten()
r_sq = np.mean(r_squares)
r_sq_sd = np.std(r_squares)
rmse = np.mean(rmses**0.5)
rmse_sd = np.std(rmses**0.5)

line11 = np.linspace(min(min(y_hats_flat),min(y_trues_flat)),
                     max(max(y_hats_flat),max(y_trues_flat)))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_trues_flat,y_hats_flat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate (mg/L)')
plt.ylabel('Predicted Nitrate (mg/L)')
plt.text(2,1,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
plt.text(2,0.5,'rmse ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
plt.legend()
plt.show()

# make better plot

# test_names = X_test.Name.reset_index(drop=True)
# test_filt = X_test.Filtered.reset_index(drop=True)

# data_out = pd.DataFrame({'y_test':y_test,'y_pred':Y_hat[:,0]})
# data_out = pd.concat([data_out,test_names,test_filt],axis=1)

# sns.set_theme(style ='ticks',font_scale = 1.25,
#               palette = 'colorblind')

# g = sns.relplot(
#     data=data_out,
#     x = 'y_test',
#     y = 'y_pred',
#     hue = 'Filtered',
#     style = 'Name',
#     s = 60
#     )

# plt.plot(line11,line11,label= '1:1 line',color = 'k',ls = 'dashed')
# plt.xlabel('Lab Measured Nitrate (mg/L)')
# plt.ylabel('Predicted Nitrate (mg/L)')
# plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))

## Phosphate (trained withough Kalera data), PLS

# Best r_sq achieved by including names and abs

param_grid = [{'n_components':np.arange(1,21)}]

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Phosphate
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])

# just absorbance values
X = abs_wq_df.loc[keep,'band_1':'band_1024']

# dummy variables
name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
filtered_dum = pd.get_dummies(abs_wq_df['Filtered'][keep])

# include name_dum
X = pd.concat([name_dum,X],axis=1)

# include filtered_dum
# X = pd.concat([filtered_dum,X],axis=1)

# include name_dum and filtered_dum
# X = pd.concat([filtered_dum,name_dum,X],axis=1)

Y = abs_wq_df.Phosphate[keep]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

num_b = 10
train_inds = np.zeros([X_train.shape[0],num_b])
test_inds = np.zeros([X_test.shape[0],num_b])
r_squares = np.zeros(num_b)
rmses = np.zeros(num_b)
errors = np.zeros([X_test.shape[0],num_b])
n_comps = np.zeros(num_b)
y_hats = np.zeros([X_test.shape[0],num_b])
y_trues = np.zeros([X_test.shape[0],num_b])

for b in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=b)
    X_test = resample(X_test,random_state = b)
    X_train = resample(X_train,random_state = b)
    y_train = resample(y_train,random_state = b)
    y_test = resample(y_test,random_state = b)
    
    train_inds[:,b]=X_train.index
    test_inds[:,b]=X_test.index

    # X_train = X_train.loc[:,'band_1':'band_1024']
    # X_test = X_test.loc[:,'band_1':'band_1024']
    pls = PLSRegression()
    clf = GridSearchCV(pls,param_grid)
    clf.fit(X_train,y_train)
    pls_opt = clf.best_estimator_
    Y_hat = pls_opt.predict(X_test)
    
    n_comp = clf.best_params_['n_components']
    r_sq = pls_opt.score(X_test,y_test)
    rmse = MSE(y_test,Y_hat)
    
    n_comps[b]=n_comp
    r_squares[b]=r_sq
    rmses[b]= rmse
    errors[:,b]=y_test-Y_hat[:,0]
    y_hats[:,b]=Y_hat[:,0]
    y_trues[:,b]=y_test

# plt.plot(Y_hat,y_test,'b.')

y_hats_flat = y_hats.flatten()
y_trues_flat = y_trues.flatten()
r_sq = np.mean(r_squares)
r_sq_sd = np.std(r_squares)
rmse = np.mean(rmses**0.5)
rmse_sd = np.std(rmses**0.5)

line11 = np.linspace(min(min(y_hats_flat),min(y_trues_flat)),
                     max(max(y_hats_flat),max(y_trues_flat)))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_trues_flat,y_hats_flat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Phosphate (mg/L)')
plt.ylabel('Predicted Phosphate (mg/L)')
plt.text(0.6,0.2,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
plt.text(0.6,0.1,'rmse ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
plt.legend()
plt.show()

# make better plot

# test_names = X_test.Name.reset_index(drop=True)
# test_filt = X_test.Filtered.reset_index(drop=True)

# data_out = pd.DataFrame({'y_test':y_test,'y_pred':Y_hat[:,0]})
# data_out = pd.concat([data_out,test_names,test_filt],axis=1)

# sns.set_theme(style ='ticks',font_scale = 1.25,
#               palette = 'colorblind')

# g = sns.relplot(
#     data=data_out,
#     x = 'y_test',
#     y = 'y_pred',
#     hue = 'Filtered',
#     style = 'Name',
#     s = 60
#     )

# plt.plot(line11,line11,label= '1:1 line',color = 'k',ls = 'dashed')
# plt.xlabel('Lab Measured Nitrate (mg/L)')
# plt.ylabel('Predicted Nitrate (mg/L)')
# plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))

## Nitrate (trained withough Kalera data), RF

# Best r_sq achieved by including names and abs

param_grid = [{'max_features':np.arange(10,120,10)}]

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Nitrate
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])

# just absorbance values
X = abs_wq_df.loc[keep,'band_1':'band_1024']
num_wavelengths = 2**7
step = 1024/num_wavelengths
wavelength_inds = np.arange(0,1024,step)
X = X.iloc[:,wavelength_inds]

# dummy variables
name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
filtered_dum = pd.get_dummies(abs_wq_df['Filtered'][keep])

# include name_dum
X = pd.concat([name_dum,X],axis=1)

# include filtered_dum
# X = pd.concat([filtered_dum,X],axis=1)

# include name_dum and filtered_dum
# X = pd.concat([filtered_dum,name_dum,X],axis=1)

Y = abs_wq_df.Nitrate[keep]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

num_b = 10
train_inds = np.zeros([X_train.shape[0],num_b])
test_inds = np.zeros([X_test.shape[0],num_b])
r_squares = np.zeros(num_b)
rmses = np.zeros(num_b)
errors = np.zeros([X_test.shape[0],num_b])
max_feats = np.zeros(num_b)
y_hats = np.zeros([X_test.shape[0],num_b])
y_trues = np.zeros([X_test.shape[0],num_b])

for b in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=b)
    X_test = resample(X_test,random_state = b)
    X_train = resample(X_train,random_state = b)
    y_train = resample(y_train,random_state = b)
    y_test = resample(y_test,random_state = b)
    
    train_inds[:,b]=X_train.index
    test_inds[:,b]=X_test.index

    # X_train = X_train.loc[:,'band_1':'band_1024']
    # X_test = X_test.loc[:,'band_1':'band_1024']
    
    rf = RandomForestRegressor()
    clf = GridSearchCV(rf,param_grid)
    clf.fit(X_train,y_train)
    rf_opt = clf.best_estimator_
    
    Y_hat = rf_opt.predict(X_test)
    
    max_feat = clf.best_params_['max_features']
    r_sq = rf_opt.score(X_test,y_test)
    rmse = MSE(y_test,Y_hat)
    
    max_feats[b]=max_feat
    r_squares[b]=r_sq
    rmses[b]= rmse
    errors[:,b]=y_test-Y_hat
    y_hats[:,b]=Y_hat
    y_trues[:,b]=y_test

# plt.plot(Y_hat,y_test,'b.')

y_hats_flat = y_hats.flatten()
y_trues_flat = y_trues.flatten()
r_sq = np.mean(r_squares)
r_sq_sd = np.std(r_squares)
rmse = np.mean(rmses**0.5)
rmse_sd = np.std(rmses**0.5)

line11 = np.linspace(min(min(y_hats_flat),min(y_trues_flat)),
                     max(max(y_hats_flat),max(y_trues_flat)))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_trues_flat,y_hats_flat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate (mg/L)')
plt.ylabel('Predicted Nitrate (mg/L)')
plt.text(2,1,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
plt.text(2,0.5,'rmse ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
plt.legend()
plt.show()

# make better plot

# test_names = X_test.Name.reset_index(drop=True)
# test_filt = X_test.Filtered.reset_index(drop=True)

# data_out = pd.DataFrame({'y_test':y_test,'y_pred':Y_hat[:,0]})
# data_out = pd.concat([data_out,test_names,test_filt],axis=1)

# sns.set_theme(style ='ticks',font_scale = 1.25,
#               palette = 'colorblind')

# g = sns.relplot(
#     data=data_out,
#     x = 'y_test',
#     y = 'y_pred',
#     hue = 'Filtered',
#     style = 'Name',
#     s = 60
#     )

# plt.plot(line11,line11,label= '1:1 line',color = 'k',ls = 'dashed')
# plt.xlabel('Lab Measured Nitrate (mg/L)')
# plt.ylabel('Predicted Nitrate (mg/L)')
# plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))

## Phosphate (trained withough Kalera data), RF

# Best r_sq achieved by including names and abs

param_grid = [{'max_features':np.arange(10,120,10)}]

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Phosphate
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])

# just absorbance values
X = abs_wq_df.loc[keep,'band_1':'band_1024']
num_wavelengths = 2**7
step = 1024/num_wavelengths
wavelength_inds = np.arange(0,1024,step)
X = X.iloc[:,wavelength_inds]

# dummy variables
name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
filtered_dum = pd.get_dummies(abs_wq_df['Filtered'][keep])

# include name_dum
X = pd.concat([name_dum,X],axis=1)

# include filtered_dum
# X = pd.concat([filtered_dum,X],axis=1)

# include name_dum and filtered_dum
# X = pd.concat([filtered_dum,name_dum,X],axis=1)

Y = abs_wq_df.Phosphate[keep]

num_b = 10
train_inds = np.zeros([X_train.shape[0],num_b])
test_inds = np.zeros([X_test.shape[0],num_b])
r_squares = np.zeros(num_b)
rmses = np.zeros(num_b)
errors = np.zeros([X_test.shape[0],num_b])
max_feats = np.zeros(num_b)
y_hats = np.zeros([X_test.shape[0],num_b])
y_trues = np.zeros([X_test.shape[0],num_b])

for b in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=b)
    X_test = resample(X_test,random_state = b)
    X_train = resample(X_train,random_state = b)
    y_train = resample(y_train,random_state = b)
    y_test = resample(y_test,random_state = b)
    
    train_inds[:,b]=X_train.index
    test_inds[:,b]=X_test.index

    # X_train = X_train.loc[:,'band_1':'band_1024']
    # X_test = X_test.loc[:,'band_1':'band_1024']
    
    rf = RandomForestRegressor()
    clf = GridSearchCV(rf,param_grid)
    clf.fit(X_train,y_train)
    rf_opt = clf.best_estimator_
    
    Y_hat = rf_opt.predict(X_test)
    
    max_feat = clf.best_params_['max_features']
    r_sq = rf_opt.score(X_test,y_test)
    rmse = MSE(y_test,Y_hat)
    
    max_feats[b]=max_feat
    r_squares[b]=r_sq
    rmses[b]= rmse
    errors[:,b]=y_test-Y_hat
    y_hats[:,b]=Y_hat
    y_trues[:,b]=y_test

# plt.plot(Y_hat,y_test,'b.')

y_hats_flat = y_hats.flatten()
y_trues_flat = y_trues.flatten()
r_sq = np.mean(r_squares)
r_sq_sd = np.std(r_squares)
rmse = np.mean(rmses**0.5)
rmse_sd = np.std(rmses**0.5)

line11 = np.linspace(min(min(y_hats_flat),min(y_trues_flat)),
                     max(max(y_hats_flat),max(y_trues_flat)))

# lr = LinearRegression().fit(Y_hat,y_test)
# linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_trues_flat,y_hats_flat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Phosphate (mg/L)')
plt.ylabel('Predicted Phosphate (mg/L)')
plt.text(0.6,0.2,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
plt.text(0.6,0.1,'rmse ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
plt.legend()
plt.show()

# make better plot

# test_names = X_test.Name.reset_index(drop=True)
# test_filt = X_test.Filtered.reset_index(drop=True)

# data_out = pd.DataFrame({'y_test':y_test,'y_pred':Y_hat[:,0]})
# data_out = pd.concat([data_out,test_names,test_filt],axis=1)

# sns.set_theme(style ='ticks',font_scale = 1.25,
#               palette = 'colorblind')

# g = sns.relplot(
#     data=data_out,
#     x = 'y_test',
#     y = 'y_pred',
#     hue = 'Filtered',
#     style = 'Name',
#     s = 60
#     )

# plt.plot(line11,line11,label= '1:1 line',color = 'k',ls = 'dashed')
# plt.xlabel('Lab Measured Nitrate (mg/L)')
# plt.ylabel('Predicted Nitrate (mg/L)')
# plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))


##### Simple Linear Regression

y = abs_wq_df['Nitrate-N']
x = abs_wq_df['band_169'].to_numpy()

lr = LinearRegression().fit(x.reshape(-1, 1),y)

y_hat = lr.predict(x.reshape(-1,1))

r_sq = r2_score(y,y_hat)
rmse = MSE(y,y_hat,squared=False)

line11 = np.linspace(min(min(y_hat),min(y)),
                     max(max(y_hat),max(y)))

plt.plot(y,y_hat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate-N (mg/L)')
plt.ylabel('Predicted Nitrate-N (mg/L)')
# plt.text(0.6,0.2,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
# plt.text(0.6,0.1,'rmse ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
plt.legend()
plt.show()

plt.plot(x,y,'o',markersize = 4, label = 'predictions')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Absorbance at 264 nm')
plt.ylabel('Nitrate-N (mg/L)')
# plt.text(0.6,0.2,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
# plt.text(0.6,0.1,'rmse ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
# plt.legend()
plt.show()

##### at 231.45 nm

y = abs_wq_df['Nitrate-N']
x = abs_wq_df['band_101'].to_numpy()

lr = LinearRegression().fit(x.reshape(-1, 1),y)

y_hat = lr.predict(x.reshape(-1,1))

r_sq = r2_score(y,y_hat)
rmse = MSE(y,y_hat,squared=False)

line11 = np.linspace(min(min(y_hat),min(y)),
                     max(max(y_hat),max(y)))

plt.plot(y,y_hat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate-N (mg/L)')
plt.ylabel('Predicted Nitrate-N (mg/L)')
# plt.text(0.6,0.2,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
# plt.text(0.6,0.1,'rmse ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
plt.legend()
plt.show()

plt.plot(x,y,'o',markersize = 4, label = 'predictions')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Absorbance at 231 nm')
plt.ylabel('Nitrate-N (mg/L)')
# plt.text(0.6,0.2,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
# plt.text(0.6,0.1,'rmse ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
# plt.legend()
plt.show()

plt.plot(1/x,np.log(y),'o',markersize = 4, label = 'predictions')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Absorbance at 231 nm')
plt.ylabel('Nitrate-N (mg/L)')
# plt.text(0.6,0.2,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
# plt.text(0.6,0.1,'rmse ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
# plt.legend()
plt.show()

#### boxplot of wq data

sns.boxplot(x='Species',y='Value',data=wq_df.loc[(wq_df['Species']=='Nitrate-N')|
            (wq_df['Species']=='Phosphorus')| (wq_df['Species']=='Potassium'),:])
