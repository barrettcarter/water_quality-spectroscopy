# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:14:36 2021

@author: aditya01
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
from keras.models import Sequential
# from keras.layers import Conv1D, MaxPool1D, AveragePooling1D, Dense, Flatten, Dropout
from keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, Dropout
from keras.regularizers import l2
# from keras.constraints import maxnorm
from keras.optimizers import Adam
# import skopt
# from scipy.signal import savgol_filter
from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score

#%%
################################################################################ DEFAULTS

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
spectra_path = os.path.join(path_to_wqs,'Hydroponics/inputs/')
abs_wq_fn = 'abs-wq_HNSr_df.csv'
spectra_path = os.path.join(spectra_path,abs_wq_fn)
os.path.exists(spectra_path)
np.random.seed(7)

#%%

inVars=['Nitrate-N', 'Ammonium-N', 'Phosphorus', 'Potassium', 'Calcium',
       'Magnesium', 'Sulfate', 'Boron', 'Zinc', 'Manganese', 'Iron', 'Copper',
       'Molybdenum', 'pH']

iEpochs=5000

augmentRatio=7

df=pd.read_csv(spectra_path)

specCols=[x for x in df.columns if x.startswith('band_')]

df=df.dropna(axis=0)

ds_x=df.loc[:,specCols].values


#%%
################################################################################ FUNCTIONS

add_dimension = lambda ds_x: np.reshape(ds_x, (ds_x.shape[0], 1, ds_x.shape[1]))

# I'm not entirely sure if augmentation is well set-up, but the models don't work without this
def augment(x, betashift, slopeshift, multishift):
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    offset = slope*(axis) + beta - axis - slope/2 + 0.5
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1
    return multi*x + offset

def predict(models, input):
    output = np.zeros((input.shape[0]))
    for model in models:
        output += model.predict(input).flatten()
    return output / len(models)

#%%
################################################################################ PROCESS

scaler_x = MinMaxScaler()
ds_x = scaler_x.fit_transform(ds_x)
scaler_y = MinMaxScaler()
#ds_y = scaler_y.fit_transform(ds_y)

"""## Smoothing
Using Savitzky-Golay
"""
# window = 5
# degree = 3

# #x = np.array(list(map(lambda s: float(s[4:].replace('_', '.')), df.columns[6:])))
# x=WVLs
# ds_x_smooth = np.array(list(map(lambda y: savgol_filter((x, y), window, degree)[1], ds_x)))
ds_x_smooth=ds_x 
"""COMMENTED OUT"""

# plt.figure(figsize=(12, 8))
# plt.plot(x, ds_x[2])
# plt.plot(x, ds_x_smooth[2])
# plt.legend(['Original', 'Smooth'])
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflectance (Normalized)')
# plt.title(f'Savitzky-Golay Smoothing (Window Size={window}, Degree={degree})')
# plt.show()


""""DROPPING OUT EVERY N'th WAVELENGTH"""
n_drop = 2
ds_x_smooth = ds_x_smooth[:,range(0,ds_x_smooth.shape[1],n_drop)]

plt.figure(figsize=(12, 8))
for i in range(ds_x_smooth.shape[0]):
    plt.plot(ds_x_smooth[i])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (Normalized)')
plt.show()

v = inVars[12]
# v="Molybdenum"
for v in inVars:
    inVar=v
    ds_y=df.loc[df[inVar]>0,inVar].values
    # ds_y=df[[inVar]][df[[inVar]]>0].values
    ds_y.shape=(len(ds_y),1)
    train_y=ds_y
    
    """## Augmentation TEST"""
    
    # x = train_x[0:1]
    # x = np.repeat(x, 10, 0)
    # x_aug = augment(x, .1, .3, .1)
    
    # plt.figure(figsize=(8, 6))
    # plt.plot(x_aug.T)
    # plt.plot(x.T, lw=3, c='b')
    # #plt.ylim(0, 1)
    # plt.xlabel('Wavelength Index')
    # plt.ylabel('Reflectance (Normalized)')
    # plt.title('Data Augmentation Sample (Exagerated Shifts)')
    
    """## Augmentation"""
    
    train_x=ds_x_smooth[df[inVar]>0,:]
    multiplier = augmentRatio
    train_x = train_x.repeat(multiplier, axis=0)
    train_x = augment(train_x, .01, .01, .01)
    
    train_y = train_y.repeat(multiplier, axis=0)
    
    """## Train/Val Split"""
    print('Full set:',train_x.shape)
    
    train_x, val_x, train_y, val_y = train_test_split(train_x,train_y, test_size=.25,
                                                      random_state= 1)
    train_x, val_e_x, train_y, val_e_y = train_test_split(train_x,train_y, test_size=.20,
                                                          random_state=1)
    
    print('Train set:',train_x.shape)
    print('Validation set:',val_x.shape)
    print('External set:',val_e_x.shape)
    
    """## Add Dimension
    """
    train_x = add_dimension(train_x)
    val_x = add_dimension(val_x)
    val_e_x = add_dimension(val_e_x)
    
    """# Training Creation
    
    ## LeNet 5
    """
    
    drop = 0.05
    reg = .01
    
    lenet = Sequential()
    lenet.add(Conv1D(6, 1,activation='relu',input_shape=(1, train_x.shape[2])))
    lenet.add(Dropout(drop))
    lenet.add(AveragePooling1D(1))
    lenet.add(Conv1D(16, 1, activation='relu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))
    lenet.add(Dropout(drop))
    lenet.add(Flatten())
    lenet.add(Dense(120, activation='relu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))
    lenet.add(Dropout(drop))
    lenet.add(Dense(84, activation='relu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))
    lenet.add(Dropout(drop))
    lenet.add(Dense(train_y.shape[1]))
    lenet.compile(loss='mean_squared_error', optimizer=Adam(.001))
    history = lenet.fit(
        train_x, train_y,
        epochs=iEpochs, batch_size=int(1e10),
        verbose=2,
        validation_data=(val_x, val_y),
    )
    
    ################################################################################ CHECK FITS
    
    plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.ylim((0, .05))
    # plt.ylim(0, 0.05)
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    
    ################################################################################ PREDICT
    
    val_i_y = predict([lenet], val_e_x)
    val_v_y = predict([lenet], val_x)
    val_m_y = predict([lenet], train_x)
    
    mod_x, mod_y = (np.expand_dims(train_y.flatten(), axis=1),np.expand_dims(val_m_y.flatten(), axis=1))
    regr = linear_model.LinearRegression()
    regr.fit(mod_x, mod_y)
    y_pred = regr.predict(mod_x)
    print('Model calibration',r2_score(mod_y, y_pred))
    
    int_x, int_y = (np.expand_dims(val_y.flatten(), axis=1),np.expand_dims(val_v_y.flatten(), axis=1))
    regr = linear_model.LinearRegression()
    regr.fit(int_x, int_y)
    y_pred = regr.predict(int_x)
    print('Internal validation',r2_score(int_y, y_pred))
    
    ext_x, ext_y = (np.expand_dims(val_e_y.flatten(), axis=1),np.expand_dims(val_i_y.flatten(), axis=1))
    regr = linear_model.LinearRegression()
    regr.fit(ext_x, ext_y)
    y_pred = regr.predict(ext_x)
    
    print('External validation',r2_score(ext_x, ext_y))
    plt.figure(figsize=(12,8))
    plt.scatter(ext_x, ext_y, c='red',alpha=0.5,s=200)
    plt.scatter(int_x, int_y, c='blue',alpha=0.5,s=200)
    plt.scatter(mod_x, mod_y, c='none',alpha=0.5,s=200,edgecolors='black')
    #plt.plot(x, y_pred, linewidth=3)
    plt.plot([np.min(mod_x),np.max(mod_x)], [np.min(mod_x),np.max(mod_x)], color='k', linestyle='--', linewidth=2)
    
    #plt.ylim(0, 1)
    #plt.xlim(0, 1)
    plt.xlabel('Observed',fontsize=14)
    plt.ylabel('Predicted',fontsize=14)
    plt.title(inVar+f' model (R^2={round(r2_score(ext_x, ext_y), 2)})',fontsize=14)
    plt.show()



