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

## Nitrate

# Best r_sq achieved by not including name or filtered with abs

param_grid = [{'n_components':np.arange(1,21)}]

keep = (abs_wq_df['Name']!='kalera1')&(abs_wq_df['Name']!='kalera2')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
# X = abs_wq_df.loc[:,'band_1':'band_1024']
# Y = abs_wq_df.Nitrate.to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'])
# filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
X = abs_wq_df.loc[keep,'band_1':'band_1024'].to_numpy()
Y = abs_wq_df.Nitrate[keep].to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
# X = pd.concat([name_dum,X],axis=1).to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
pls = PLSRegression()
clf = GridSearchCV(pls,param_grid)
clf.fit(X_train,y_train)
n_comp = clf.best_params_['n_components']
pls_opt = clf.best_estimator_
Y_hat = pls_opt.predict(X_test)

r_sq = pls_opt.score(X_test,y_test)

plt.plot(Y_hat,y_test,'b.')

line11 = np.linspace(min(y_test),max(y_test))

plt.plot(Y_hat,y_test,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
plt.xlabel('Predicted Nitrate')
plt.ylabel('True Nitrate')
plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.show()

X_kal = abs_wq_df.loc[keep==False,'band_1':'band_1024'].to_numpy()
y_kal = abs_wq_df.Nitrate[keep==False].to_numpy()
Y_hat = pls_opt.predict(X_kal)

r_sq = pls_opt.score(X_kal,y_kal)

plt.plot(Y_hat,y_kal,'b.')

line11 = np.linspace(min(y_kal),max(y_kal))

plt.plot(Y_hat,y_kal,'o',markersize = 4, label = 'predictions')
plt.plot(line11,line11,label= '1:1 line')
plt.xlabel('Predicted Nitrate')
plt.ylabel('True Nitrate')
plt.text(160,210,r'$r^2 =$'+str(np.round(r_sq,3)))
plt.show()

coefs = pls.coef_

plt.plot(coefs[0:200])

## Phosphate

# best r_sq when just including names with abs

param_grid = [{'n_components':np.arange(1,21)}]

keep = (abs_wq_df['Name']!='swb')
# keep = abs_wq_df['Name'].isin(['hogdn','hat'])
# X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
X = abs_wq_df.loc[:,'band_1':'band_1024']
Y = abs_wq_df.Phosphate.to_numpy()
name_dum = pd.get_dummies(abs_wq_df['Name'])
filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
# X = abs_wq_df.loc[keep,'band_1':'band_1024']
# Y = abs_wq_df.Phosphate[keep].to_numpy()
# name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
# X = pd.concat([name_dum,filtered_dum,X],axis=1).to_numpy()
# X = pd.concat([filtered_dum,X],axis=1).to_numpy()
X = pd.concat([name_dum,X],axis=1).to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
pls = PLSRegression()
clf = GridSearchCV(pls,param_grid)
clf.fit(X_train,y_train)
print(clf.best_params_)
n_comp = clf.best_params_['n_components']
pls_opt = clf.best_estimator_
Y_hat = pls_opt.predict(X_test)

r_sq = pls_opt.score(X_test,y_test)

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