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

# IMPORTANT: Don't have the baseDir and saveDir be the same
user = os.getlogin() 
abs_df_dir='C:/Users/'+user+'/OneDrive/Documents/Data/Inputs/abs/'
wq_df_dir='C:/Users/'+user+'/OneDrive/Documents/Data/Inputs/wq/'

# wq_df_fn='wq_aj_df.csv'
wq_df_fn = 'wq_tot_df.csv'
# wq_kal_df_fn = 'wq_kalera_df.csv'
abs_df_fn = 'abs_df_u2d.csv'

# Bring in data
abs_df=pd.read_csv(abs_df_dir+abs_df_fn)
wq_df=pd.read_csv(wq_df_dir+wq_df_fn)
# wq_kal_df = pd.read_csv(wq_df_dir+wq_kal_df_fn)

# Make names consistent formatting

wq_df.Name = wq_df.Name.apply(lambda x: x.lower())
wq_df.Name[wq_df.Name=='hognw16']='hogup' # this makes a warning but is okay
wq_df.Name.unique()
# Make species names consistent

wq_df.Species.unique()

wq_df.Species[wq_df.Species=='Ammonia-Nitrogen']='Ammonia (N)'
wq_df.Species[wq_df.Species=='Phosphorus']='Orthophosphate (P)'
wq_df.Species[wq_df.Species=='Nitrate-Nitrogen']='Nitrate-Nitrite (N)'

# Get rid of wq data from 11/5/2020 (mix-up with sample dates)

wq_df = wq_df.loc[wq_df.Date_col!='11/5/2020',:]
wq_df=wq_df.reset_index(drop = True)

# Makes dates match

wq_dates = wq_df.Date_col.unique()
abs_dates = abs_df.Date_col.unique()
wq_df['Date_col'][wq_df.Date_col=='8/14/2020']='8/13/2020'
wq_df['Date_col'][wq_df.Date_col=='8/24/2020']='8/25/2020'
wq_dates = wq_df.Date_col.unique()

# Create sample IDs for combining two dataframes

wq_df['ID']=wq_df.Name+wq_df.Date_col

# concs_ind = range(int((wq_df.shape[0]-1)/3))
# concs_df = wq_df.pivot(index = range(268),columns = 'Species',values = 'Conc')

aw_df_cols = ['Ammonium','Phosphate','Nitrate','ID']+abs_df.columns.tolist()
abs_wq_df = abs_df
abs_wq_df['Ammonium'] = -0.1
abs_wq_df['Phosphate'] = -0.1
abs_wq_df['Nitrate'] = -0.1
abs_wq_df['ID']=abs_wq_df.Name+abs_wq_df.Date_col

abs_wq_df = abs_wq_df[aw_df_cols]

for wq_row in range(wq_df.shape[0]):
    for abs_row in range(abs_wq_df.shape[0]):
        if wq_df.ID[wq_row]==abs_wq_df.ID[abs_row]:
            if wq_df.Species[wq_row] == 'Ammonia (N)':
                abs_wq_df.Ammonium[abs_row]=wq_df.Conc[wq_row]
            if wq_df.Species[wq_row] == 'Nitrate-Nitrite (N)':
                abs_wq_df.Nitrate[abs_row]=wq_df.Conc[wq_row]
            if wq_df.Species[wq_row] == 'Orthophosphate (P)':
                abs_wq_df.Phosphate[abs_row]=wq_df.Conc[wq_row]
            
abs_wq_df = abs_wq_df.loc[abs_wq_df.Ammonium>0,:]     
            
# trying out PLS

# Nitrate
X = abs_wq_df.loc[abs_wq_df.Name=='swb','band_1':'band_1024']
X = X.to_numpy()
Y = abs_wq_df.Nitrate[abs_wq_df.Name=='swb'].to_numpy()
pls = PLSRegression(n_components = 10)
pls.fit(X,Y)
Y_hat = pls.predict(X)

r_sq = pls.score(X,Y)

plt.plot(Y_hat,Y,'b.')

# with test and train data

# Nitrate

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Nitrate.to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
X = abs_wq_df.loc[keep,'band_1':'band_1024']
Y = abs_wq_df.Nitrate[keep].to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
# X = pd.concat([name_dum,filtered_dum,X],axis=1).to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
pls = PLSRegression(n_components = 10)
pls.fit(X_train,y_train)
Y_hat = pls.predict(X_test)

r_sq = pls.score(X_test,y_test)

plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(y_test),max(y_test))

plt.plot(Y_hat,y_test,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
plt.xlabel('Predicted Nitrate')
plt.ylabel('True Nitrate')
plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.show()

coefs = pls.coef_

plt.plot(coefs[0:200])

# Phosphate
    
keep = (abs_wq_df['Name']!='swb')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
X = abs_wq_df.loc[:,'band_1':'band_1024']
Y = abs_wq_df.Phosphate.to_numpy()
name_dum = pd.get_dummies(abs_wq_df['Name'])
filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
# X = abs_wq_df.loc[keep,'band_1':'band_1024']
# Y = abs_wq_df.Phosphate[keep].to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
X = pd.concat([name_dum,filtered_dum,X],axis=1).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
pls = PLSRegression(n_components = 10)
pls.fit(X_train,y_train)
Y_hat = pls.predict(X_test)

r_sq = pls.score(X_test,y_test)

plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(y_test),max(y_test))

plt.plot(Y_hat,y_test,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
plt.xlabel('Predicted Phosphate')
plt.ylabel('True Phosphate')
plt.text(0.2,0.8,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.show()

coefs = pls.coef_

plt.plot(coefs[0:200])

# # separate between filtered and unfiltered

# X = abs_wq_df.loc[abs_wq_df.Filtered==True,'band_1':'band_1024']
# X = X.to_numpy()
# Y = abs_wq_df.loc[abs_wq_df.Filtered==True,'Phosphate']
# Y = Y.to_numpy()

# X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
# pls = PLSRegression(n_components = 10)
# pls.fit(X_train,y_train)
# Y_hat = pls.predict(X_test)

# r_sq = pls.score(X_test,y_test)

# plt.plot(Y_hat,y_test,'b.')

# #not good for filtered samples

# X = abs_wq_df.loc[abs_wq_df.Filtered==False,'band_1':'band_1024']
# X = X.to_numpy()
# Y = abs_wq_df.loc[abs_wq_df.Filtered==False,'Phosphate']
# Y = Y.to_numpy()

# X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
# pls = PLSRegression(n_components = 5)
# pls.fit(X_train,y_train)
# Y_hat = pls.predict(X_test)

# r_sq = pls.score(X_test,y_test)

# plt.plot(Y_hat,y_test,'b.')
######################################################
### Tuning the models

## Nitrate (trained withough Kalera data, absorbance only)
## PlSR

# Best r_sq achieved by not including name or filtered with abs

param_grid = [{'n_components':np.arange(1,8)}]

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Nitrate.to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
X = abs_wq_df.loc[keep,:]
Y = abs_wq_df.Nitrate[keep].to_numpy()
name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
X = pd.concat([name_dum,X],axis=1)
# X = name_dum

name_dum_cols = name_dum.columns.to_numpy()
bands = X.loc[:,'band_1':'band_1024'].columns.to_numpy()
X_cols = np.concatenate((name_dum_cols,bands))

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
test_names = X_test.Name.reset_index(drop=True)
test_filt = X_test.Filtered.reset_index(drop=True)

X_train = X_train.loc[:,X_cols]
X_test = X_test.loc[:,X_cols]

# X_train = X_train.loc[:,'band_1':'band_1024']
# X_test = X_test.loc[:,'band_1':'band_1024']
pls = PLSRegression()
clf = GridSearchCV(pls,param_grid)
clf.fit(X_train,y_train)
n_comp = clf.best_params_['n_components']
pls_opt = clf.best_estimator_
Y_hat = pls_opt.predict(X_test)

r_sq = pls_opt.score(X_test,y_test)

# plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(np.concatenate((y_test,Y_hat[:,0]))),
                     max(np.concatenate((y_test,Y_hat[:,0]))))

lr = LinearRegression().fit(Y_hat,y_test)
linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate (mg/L)')
plt.ylabel('Predicted Nitrate (mg/L)')
plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.legend()
plt.show()

# make better plot
data_out = pd.DataFrame({'y_test':y_test,'y_pred':Y_hat[:,0]})
data_out = pd.concat([data_out,test_names,test_filt],axis=1)
data_out.rename(columns={'Name':'Site ID'},inplace=True)

sns.set_theme(style ='ticks',font_scale = 1.8,
              palette = 'colorblind')

g = sns.relplot(
    data=data_out,
    x = 'y_test',
    y = 'y_pred',
    hue = 'Site ID',
    style = 'Site ID',
    s = 100
    )

plt.plot(line11,line11,label= '1:1 line',color = 'k',ls = 'dashed')
plt.xlabel('Lab Measured Nitrate (mg-N/L)')
plt.ylabel('Predicted Nitrate (mg-N/L)')
plt.text(1.5,0,r'$R^2 =$'+str(np.round(r_sq,3)))

# determine r2 for sites otherthan swb

r2_score(data_out.y_test[data_out.Name!='swb'],
         data_out.y_pred[data_out.Name!='swb'])

# this r2 is less than that for when the model is trained only on these sites

# predicting concs. of Kalera data with model calibrated with AJ data
# This does not work well.

# X_kal = abs_wq_df.loc[keep==False,'band_1':'band_1024'].to_numpy()
# y_kal = abs_wq_df.Nitrate[keep==False].to_numpy()
# Y_hat = pls_opt.predict(X_kal)

# r_sq = pls_opt.score(X_kal,y_kal)

# plt.plot(Y_hat,y_kal,'b.')

# line11 = np.linspace(min(y_kal),max(y_kal))

# plt.plot(Y_hat,y_kal,'o',markersize = 4, label = 'predictions')
# plt.plot(line11,line11,label= '1:1 line')
# plt.xlabel('Predicted Nitrate')
# plt.ylabel('True Nitrate')
# plt.text(160,210,r'$r^2 =$'+str(np.round(r_sq,3)))
# plt.show()

# coefs = pls.coef_

# plt.plot(coefs[0:200])

# Nitrate without swb
# best results obtained with abs and name

param_grid = [{'n_components':np.arange(1,21)}]

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')\
    &(abs_wq_df['Name']!='swb')

# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Nitrate.to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
# X = abs_wq_df.loc[keep,'band_1':'band_1024'].to_numpy()
X = abs_wq_df.loc[keep,'band_1':'band_1024'] # if needed to be concatenated
Y = abs_wq_df.Nitrate[keep].to_numpy()
name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
filtered_dum = pd.get_dummies(abs_wq_df['Filtered'][keep])
# X = pd.concat([name_dum,X],axis=1).to_numpy()
X = pd.concat([name_dum,X],axis=1).to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
pls = PLSRegression()
clf = GridSearchCV(pls,param_grid)
clf.fit(X_train,y_train)
n_comp = clf.best_params_['n_components']
pls_opt = clf.best_estimator_
Y_hat = pls_opt.predict(X_test)

r_sq = pls_opt.score(X_test,y_test)

plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(np.concatenate((y_test,Y_hat[:,0]))),
                     max(np.concatenate((y_test,Y_hat[:,0]))))

lr = LinearRegression().fit(Y_hat,y_test)
linelr = lr.predict(line11.reshape(-1,1))

plt.plot(Y_hat,y_test,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Predicted Nitrate (mg/L)')
plt.ylabel('True Nitrate (mg/L)')
plt.text(0.5,0.1,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.legend()
plt.show()

## Random Forest (nitrate)

# Best r_sq achieved with...
# max_features ~70 seems to do best, but this varies a lot

param_grid = [{'max_features':np.arange(10,110,10)}]
# param_grid = [{'max_features':np.arange(60,80,2)}]

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Nitrate.to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
X = abs_wq_df.loc[keep,'band_1':'band_1024']
Y = abs_wq_df.Nitrate[keep].to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
# X = pd.concat([name_dum,X],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
# test_names = X_test.Name.reset_index(drop=True)
# test_filt = X_test.Filtered.reset_index(drop=True)

# absorbance values
num_wavelengths = 2**7
step = 1024/num_wavelengths
wavelength_inds = np.arange(0,1024,step)
X_train = X_train.iloc[:,wavelength_inds]
X_test = X_test.iloc[:,wavelength_inds]

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

# plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(np.concatenate((y_test,Y_hat))),
                     max(np.concatenate((y_test,Y_hat))))

lr = LinearRegression().fit(Y_hat,y_test)
linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate (mg/L)')
plt.ylabel('Predicted Nitrate (mg/L)')
plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))
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

## Phosphate without Kalera data

# best r_sq whe including abs, names and filtration
# good results also obtained with just names

## Nitrate (trained withough Kalera data, absorbance only)
## PlSR

# Best r_sq achieved by not including name or filtered with abs

param_grid = [{'n_components':np.arange(1,8)}]

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Nitrate.to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
X = abs_wq_df.loc[keep,:]
Y = abs_wq_df.Phosphate[keep].to_numpy()
name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
X = pd.concat([name_dum,X],axis=1)
# X = name_dum

name_dum_cols = name_dum.columns.to_numpy()
bands = X.loc[:,'band_1':'band_1024'].columns.to_numpy()
X_cols = np.concatenate((name_dum_cols,bands))

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
test_names = X_test.Name.reset_index(drop=True)
test_filt = X_test.Filtered.reset_index(drop=True)

X_train = X_train.loc[:,X_cols]
X_test = X_test.loc[:,X_cols]

# X_train = X_train.loc[:,'band_1':'band_1024']
# X_test = X_test.loc[:,'band_1':'band_1024']
pls = PLSRegression()
clf = GridSearchCV(pls,param_grid)
clf.fit(X_train,y_train)
n_comp = clf.best_params_['n_components']
pls_opt = clf.best_estimator_
Y_hat = pls_opt.predict(X_test)

r_sq = pls_opt.score(X_test,y_test)

# plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(np.concatenate((y_test,Y_hat[:,0]))),
                     max(np.concatenate((y_test,Y_hat[:,0]))))

lr = LinearRegression().fit(Y_hat,y_test)
linelr = lr.predict(line11.reshape(-1,1))

plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Phosphate (mg/L)')
plt.ylabel('Predicted Phosphate (mg/L)')
plt.text(0.1,0.7,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.legend()
plt.show()

# make better plot
data_out = pd.DataFrame({'y_test':y_test,'y_pred':Y_hat[:,0]})
data_out = pd.concat([data_out,test_names,test_filt],axis=1)
data_out.rename(columns={'Name':'Site ID'},inplace=True)

sns.set_theme(style ='ticks',font_scale = 1.8,
              palette = 'colorblind')

g = sns.relplot(
    data=data_out,
    x = 'y_test',
    y = 'y_pred',
    hue = 'Site ID',
    style = 'Site ID',
    s = 100
    )

plt.plot(line11,line11,label= '1:1 line',color = 'k',ls = 'dashed')
plt.xlabel('Lab Measured Phosphate (mg-P/L)')
plt.ylabel('Predicted Phosphate (mg-P/L)')
plt.text(0.5,0.075,r'$R^2 =$'+str(np.round(r_sq,3)))

coefs = pls.coef_

plt.plot(coefs[0:200])

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

sns.set_theme(style ='ticks',font_scale = 1.25,
              palette = 'colorblind')

plt.plot(y_trues_flat,y_hats_flat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Nitrate (mg-N/L)')
plt.ylabel('Predicted Nitrate (mg-N/L)')
plt.text(1.5,0.5,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
plt.text(1.5,0,'RMSE ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
# plt.legend()
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

sns.set_theme(style ='ticks',font_scale = 1.25,
              palette = 'colorblind')

plt.plot(y_trues_flat,y_hats_flat,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
# plt.plot(line11,linelr,label = 'regression line')
plt.xlabel('Lab Measured Phosphate (mg-P/L)')
plt.ylabel('Predicted Phosphate (mg-P/L)')
plt.text(0.45,0.1,r'$r^2 =$'+str(np.round(r_sq,3))+r'$\pm$'+str(np.round(r_sq_sd,3)))
plt.text(0.45,0,'RMSE ='+str(np.round(rmse,3))+r'$\pm$'+str(np.round(rmse_sd,3)))
# plt.legend()
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
