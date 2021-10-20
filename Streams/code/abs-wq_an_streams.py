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
abs_df_dir=os.path.join(path_to_wqs,'Data/spectra/')
wq_df_dir=os.path.join(path_to_wqs,'Streams/inputs/water_quality/')


wq_df_fn = 'wq_streams_aj_df.csv'
wq_df_fn2 = 'wq_streams_arl_df.csv'
wq_codes_fn = 'ARL_codes.csv'
abs_df_fn = 'abs_df_u2d.csv'

# Bring in data
abs_df=pd.read_csv(abs_df_dir+abs_df_fn)
wq_df=pd.read_csv(wq_df_dir+wq_df_fn)
wq_df2=pd.read_csv(wq_df_dir+wq_df_fn2)
wq_codes = pd.read_csv(wq_df_dir+wq_codes_fn)

#%%

### Some data wrangling

# Select only stream samples

stream_names = wq_df['Name'].unique()

stream_abs = []

for n in abs_df.Name:
    stream_abs.append(any(stream_names==n))

abs_df = abs_df.iloc[stream_abs,:]

# Clean up wq_df2

wq_df2 = wq_df2.loc[pd.notna(wq_df2['ARL_code']),:]
wq_df2.reset_index(drop = True,inplace = True)

# make ID column

wq_df['ID']=wq_df['Name']+wq_df['Date_col']

#%%

### Make dataframe with absorbances and water quality

species = wq_df.Species.unique()
species = np.concatenate((species,['TKN','TP','ID']))
species = np.delete(species,0) #get rid of Ammonium
aw_df_cols = np.append(species,abs_df.columns)
abs_wq_df = abs_df

abs_wq_df.loc[:,species] = -0.1
abs_wq_df['ID']=abs_wq_df.Name+abs_wq_df.Date_col
abs_wq_df = abs_wq_df[aw_df_cols]
abs_wq_df.reset_index(drop = True,inplace = True)

species = np.delete(species,len(species)-1) #get rid of ID

for wq_row in range(wq_df.shape[0]):
    for abs_row in range(abs_wq_df.shape[0]):
        if wq_df.ID[wq_row]==abs_wq_df.ID[abs_row]:
            for s in species:
                if wq_df.Species[wq_row] == s:
                    abs_wq_df.loc[abs_row,s]=wq_df.Conc[wq_row]
                             
#%%
#################################################################


### Tuning the models

## Nitrate
## PlSR

for rs in range(10):

    param_grid = [{'n_components':np.arange(1,20)}]
    
    # keep = abs_wq_df['Name'].isin(['hogdn','hat'])
    # X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
    Y = abs_wq_df['Nitrate-N']
    keep = Y>0
    X = abs_wq_df.loc[keep,'band_1':'band_1024']
    Y = Y[keep]
    
    # name_dum = pd.get_dummies(abs_wq_df['Name'])
    # filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
    # X = abs_wq_df.loc[keep,:]
    # Y = abs_wq_df.Nitrate[keep].to_numpy()
    # name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
    # X = pd.concat([name_dum,X],axis=1).to_numpy()
    # X = name_dum
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=rs,
                                                        test_size = 0.3)
    # test_names = X_test.Name.reset_index(drop=True)
    # test_filt = X_test.Filtered.reset_index(drop=True)
    
    # X_train = X_train.loc[:,'band_1':'band_1024']
    # X_test = X_test.loc[:,'band_1':'band_1024']
    pls = PLSRegression()
    clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_absolute_error')
    # clf = GridSearchCV(pls,param_grid,scoring = 'r2')
    # clf.cv_results_
    # clf = GridSearchCV(pls,param_grid)
    clf.fit(X_train,y_train)
    n_comp = clf.best_params_['n_components']
    print('Number of components:\t'+str(n_comp))
    pls_opt = clf.best_estimator_
    Y_hat = pls_opt.predict(X_test)
    Y_hat_train = pls_opt.predict(X_train)
    
    r_sq = pls_opt.score(X_test,y_test)
    print('Test r-squared value:\t'+str(round(r_sq,3)))
    
    r_sq_train = pls_opt.score(X_train,y_train)
    print('Training r-squared value:\t'+str(round(r_sq_train,3)))
    
    
    
    ### Plot results
    # plt.plot(Y_hat,y_test,'b.')
    
    line11 = np.linspace(min(np.concatenate((y_test,Y_hat[:,0]))),
                         max(np.concatenate((y_test,Y_hat[:,0]))))
    
    # lr = LinearRegression().fit(Y_hat,y_test)
    # linelr = lr.predict(line11.reshape(-1,1))
    
    plt.figure()
    plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
    plt.plot(line11,line11,label= '1:1 line')
    # plt.plot(line11,linelr,label = 'regression line')
    plt.xlabel('Lab Measured Nitrate (mg/L)')
    plt.ylabel('Predicted Nitrate (mg/L)')
    plt.text(170,200,r'$r^2 =$'+str(np.round(r_sq,3)))
    plt.title('Test Set')
    plt.legend()
    plt.show()
    
    line11 = np.linspace(min(np.concatenate((y_train,Y_hat_train[:,0]))),
                         max(np.concatenate((y_train,Y_hat_train[:,0]))))
    
    # lr = LinearRegression().fit(Y_hat,y_test)
    # linelr = lr.predict(line11.reshape(-1,1))
    
    plt.plot(y_train,Y_hat_train,'o',markersize = 4, label = 'predictions')
    plt.plot(line11,line11,label= '1:1 line')
    # plt.plot(line11,linelr,label = 'regression line')
    plt.title('Training Set')
    plt.xlabel('Lab Measured Nitrate (mg/L)')
    plt.ylabel('Predicted Nitrate (mg/L)')
    plt.text(0.8*max(line11),min(line11),r'$r^2 =$'+str(np.round(r_sq_train,3)))
    plt.legend()
    plt.show()

#%%

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

#%% 
### Create a model for every species
s = 'Molybdenum'

outputs_df = pd.DataFrame(columns= ['output','species','iteration','value']) #save outputs in dataframe

output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
                'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
                'train_rmse','test_mape','train_mape','n_comp']

variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                  'list(X_test.index)','list(X_train.index)','r_sq','r_sq_train','RMSE_test',
                  'RMSE_train','MAPE_test','MAPE_train','n_comp']

# for vn in variable_names: # this is for testing
#     print(vn+':',type(eval(vn)))


## save outputs in dictionaries
# y_hat_test_dict = dict()
# y_hat_train_dict = dict()
# # y_true_train_dict = dict()
# # y_true_test_dict = dict()
# test_ind_dict = dict()
# train_ind_dict = dict()
# test_rsq_dict = dict()
# train_rsq_dict = dict()
# test_rmse_dict = dict()
# train_rmse_dict = dict()
# test_mape_dict = dict()
# train_mape_dict = dict()
# n_comp_dict = dict()

# for s in species:
#     y_hat_test_dict[s]=[]
#     y_hat_train_dict[s] = []
#     test_ind_dict[s] = []
#     train_ind_dict[s] = []
#     test_rsq_dict[s] = []
#     train_rsq_dict[s] = []
#     test_rmse_dict[s] = []
#     train_rmse_dict[s] = []
#     test_mape_dict[s] = []
#     train_mape_dict[s] = []
#     n_comp_dict[s] = []

def write_output_df(the_output,output_name,species_name,iteration_num):
    
    if isinstance(the_output,float):
        sub_df = pd.DataFrame([[output_name,species_name,iteration_num,the_output]],
                                       columns= ['output','species','iteration','value'])
    elif isinstance(the_output,list):
        sub_df = pd.DataFrame(columns= ['output','species','iteration','value'])
        sub_df['value']=the_output
        sub_df['output']=output_name
        sub_df['species']=species_name
        sub_df['iteration']=iteration_num
    else:
        print('Error: outputs must be of type list or float')
    return(sub_df)

iteration = 1 # this is for testing


for s in species:
    Y = abs_wq_df[s]
    keep = Y>0
    X = abs_wq_df.loc[keep,'band_1':'band_1024']
    Y = Y[keep]
    
    # name_dum = pd.get_dummies(abs_wq_df['Name'])
    # filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
    # X = abs_wq_df.loc[keep,:]
    # Y = abs_wq_df.Nitrate[keep].to_numpy()
    # name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
    # X = pd.concat([name_dum,X],axis=1).to_numpy()
    # X = name_dum
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2,
                                                        test_size = 0.3)
    # test_names = X_test.Name.reset_index(drop=True)
    # test_filt = X_test.Filtered.reset_index(drop=True)
    
    # X_train = X_train.loc[:,'band_1':'band_1024']
    # X_test = X_test.loc[:,'band_1':'band_1024']
    param_grid = [{'n_components':np.arange(1,20)}]
    pls = PLSRegression()
    clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_absolute_error')
    # clf.cv_results_
    # clf = GridSearchCV(pls,param_grid)
    clf.fit(X_train,y_train)
    n_comp = float(clf.best_params_['n_components'])
    pls_opt = clf.best_estimator_
    Y_hat = list(pls_opt.predict(X_test)[:,0])
    Y_hat_train = list(pls_opt.predict(X_train)[:,0])
    
    r_sq = float(pls_opt.score(X_test,y_test))
    r_sq_train = float(pls_opt.score(X_train,y_train))
    
    MSE_test = MSE(y_test,Y_hat)
    RMSE_test = float(np.sqrt(MSE_test))
    
    MSE_train = MSE(y_train,Y_hat_train)
    RMSE_train = float(np.sqrt(MSE_train))
    
    abs_test_errors = abs(y_test-Y_hat)
    APE_test = abs_test_errors/y_test # APE = absolute percent error,decimal
    MAPE_test = float(np.mean(APE_test)*100) # this is percentage
    
    abs_train_errors = abs(y_train-Y_hat_train)
    APE_train = abs_train_errors/y_train # APE = absolute percent error,decimal
    MAPE_train = float(np.mean(APE_train)*100) # this is percentage
    
    
    # y_hat_test_dict[s].extend(list(Y_hat))
    # y_hat_train_dict[s].extend(list(Y_hat_train))
    # test_ind_dict[s].extend(list(X_test.index))
    # train_ind_dict[s].extend(list(X_train.index))
    # test_rsq_dict[s].append(float(r_sq))
    # train_rsq_dict[s].append(float(r_sq_train))
    # test_rmse_dict[s].append(float(RMSE_test))
    # train_rmse_dict[s].append(float(RMSE_train))
    # test_mape_dict[s].append(float(MAPE_test))
    # train_mape_dict[s].append(float(MAPE_train))
    # n_comp_dict[s].append(float(n_comp))
    
    # sub_df = pd.DataFrame(columns= ['output','species','iteration','value'])
    # sub_df['value']=r_sq
    # sub_df['output']='test_rsq'
    # sub_df['species']=s
    # sub_df['iteration']=iteration
        
    # sub_df = write_output_df(r_sq, 'test_rsq', s, iteration)
    # outputs_df = outputs_df.append(sub_df,ignore_index=True)
    
    for out in range(len(output_names)):
        print(out)
        sub_df = write_output_df(eval(variable_names[out]), output_names[out], s, iteration)
        outputs_df = outputs_df.append(sub_df,ignore_index=True)
    
    # plt.plot(Y_hat,y_test,'b.')
    
    # line11 = np.linspace(min(np.concatenate((y_test,Y_hat))),
    #                      max(np.concatenate((y_test,Y_hat))))
    
    # y_text1 = min(line11)+(max(line11)-min(line11))*0.05
    # y_text2 = min(line11)+(max(line11)-min(line11))*0.15
    # x_text = max(line11)-(max(line11)-min(line11))*0.3
    
    # # lr = LinearRegression().fit(Y_hat,y_test)
    # # linelr = lr.predict(line11.reshape(-1,1))
    
    # plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
    # plt.plot(line11,line11,label= '1:1 line')
    # plt.title('Test Set')
    # # plt.plot(line11,linelr,label = 'regression line')
    # plt.xlabel('Lab Measured '+s+' (mg/L)')
    # plt.ylabel('Predicted '+s+' (mg/L)')
    # plt.text(x_text,y_text1,r'$r^2 =$'+str(np.round(r_sq,3)))
    # plt.text(x_text,y_text2,r'MAPE = '+str(np.round(MAPE_test,1))+'%')
    # plt.legend()
    # plt.show()
    
    # line11 = np.linspace(min(np.concatenate((y_train,Y_hat_train))),
    #                      max(np.concatenate((y_train,Y_hat_train))))
    
    # y_text1 = min(line11)+(max(line11)-min(line11))*0.05
    # y_text2 = min(line11)+(max(line11)-min(line11))*0.15
    
    # x_text = max(line11)-(max(line11)-min(line11))*0.3
    
    # # lr = LinearRegression().fit(Y_hat,y_test)
    # # linelr = lr.predict(line11.reshape(-1,1))
    
    # plt.plot(y_train,Y_hat_train,'o',markersize = 4, label = 'predictions')
    # plt.plot(line11,line11,label= '1:1 line')
    # plt.title('Training Set')
    # # plt.plot(line11,linelr,label = 'regression line')
    # plt.xlabel('Lab Measured '+s+' (mg/L)')
    # plt.ylabel('Predicted '+s+' (mg/L)')
    # plt.text(x_text,y_text1,r'$r^2 =$'+str(np.round(r_sq_train,3)))
    # plt.text(x_text,y_text2,r'MAPE = '+str(np.round(MAPE_train,1))+'%')
    # plt.legend()
    # plt.show()
    
#%%

## make plots
    
fig, axs = plt.subplots(4,4)
fig.set_size_inches(12.8,9.6)
axs[3, 2].axis('off')
axs[3, 3].axis('off')
row = 0
col = 0
for s in species:
    y_true_train = outputs_df.loc[((outputs_df.species == s) &
                                    (outputs_df.output == 'y_true_train')),
                                   'value']
    
    y_hat_train = outputs_df.loc[((outputs_df.species == s) &
                                    (outputs_df.output == 'y_hat_train')),
                                   'value']
    
    y_true_test = outputs_df.loc[((outputs_df.species == s) &
                                    (outputs_df.output == 'y_true_test')),
                                   'value']
    
    y_hat_test = outputs_df.loc[((outputs_df.species == s) &
                                    (outputs_df.output == 'y_hat_test')),
                                   'value']
    
    line11 = np.linspace(min(np.concatenate((y_true_train,y_hat_train,
                                             y_true_test,y_hat_test))),
                          max(np.concatenate((y_true_train,y_hat_train,
                                             y_true_test,y_hat_test))))
    
    y_text1 = min(line11)+(max(line11)-min(line11))*0.05
    y_text2 = min(line11)+(max(line11)-min(line11))*0.15
    x_text = max(line11)-(max(line11)-min(line11))*0.3
    
    # lr = LinearRegression().fit(Y_hat,y_test)
    # linelr = lr.predict(line11.reshape(-1,1))
    
    # plt.plot(y_true_train,y_hat_train,'o',markersize = 4, label = 'predictions')
    # plt.plot(line11,line11,label= '1:1 line')
    # plt.title(s)
    # # plt.plot(line11,linelr,label = 'regression line')
    # plt.xlabel('Lab Measured '+s+' (mg/L)')
    # plt.ylabel('Predicted '+s+' (mg/L)')
    # plt.text(x_text,y_text1,r'$r^2 =$'+str(np.round(r_sq,3)))
    # plt.text(x_text,y_text2,r'MAPE = '+str(np.round(MAPE_test,1))+'%')
    # plt.legend()
    # plt.show()
    
    axs[row,col].plot(y_true_train,y_hat_train,'o',markersize = 4, label = 'training set')
    axs[row,col].plot(y_true_test,y_hat_test,'o',markersize = 4, label = 'test set')
    axs[row,col].plot(line11,line11,'k--',label= '1:1 line')
    axs[row,col].set_title(s)
    axs[row,col].legend()
    axs[row,col].get_xaxis().set_visible(False)
    if col == 3:
        col = 0
        row += 1
    else:
        col +=1

#%%
fig,axs = plt.subplots(2)
fig.set_size_inches(12.8,9.6)
for plots in [0,1]:
    axs[plots].plot([1,2],[2,4])

#%%
my_dict = {}
my_dict.values

#%%
my_arr = np.empty((4,4))
my_arr[:,2] = [1,2,3,4]
my_arr[:,1] = 'hi'

#%%

## Random Forest (nitrate)

# Best r_sq achieved with...
# max_features ~70 seems to do best, but this varies a lot

param_grid = [{'max_features':np.arange(10,110,10)}]
# param_grid = [{'max_features':np.arange(60,80,2)}]

# keep = (abs_wq_df['Name']!='HNSr1')&(abs_wq_df['Name']!='HNSr2')
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

keep = (abs_wq_df['Name']!='HNSr1')&(abs_wq_df['Name']!='HNSr2')
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

## Nitrate (trained withough HNSr data), PLS

# Best r_sq achieved by including names and abs

param_grid = [{'n_components':np.arange(1,21)}]

keep = (abs_wq_df['Name']!='HNSr1')&(abs_wq_df['Name']!='HNSr2')
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

## Phosphate (trained withough HNSr data), PLS

# Best r_sq achieved by including names and abs

param_grid = [{'n_components':np.arange(1,21)}]

keep = (abs_wq_df['Name']!='HNSr1')&(abs_wq_df['Name']!='HNSr2')
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

## Nitrate (trained withough HNSr data), RF

# Best r_sq achieved by including names and abs

param_grid = [{'max_features':np.arange(10,120,10)}]

keep = (abs_wq_df['Name']!='HNSr1')&(abs_wq_df['Name']!='HNSr2')
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

## Phosphate (trained withough HNSr data), RF

# Best r_sq achieved by including names and abs

param_grid = [{'max_features':np.arange(10,120,10)}]

keep = (abs_wq_df['Name']!='HNSr1')&(abs_wq_df['Name']!='HNSr2')
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
