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

from keras.models import Sequential
# from keras.layers import Conv1D, MaxPool1D, AveragePooling1D, Dense, Flatten, Dropout
from keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, Dropout
from keras.regularizers import l2
# from keras.constraints import maxnorm
from keras.optimizers import Adam

#%% Define class for pre-processing data and training neural network.

def augment(x, betashift, slopeshift, multishift):
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    offset = slope*(axis) + beta - axis - slope/2 + 0.5
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1
    return(multi*x + offset)

def add_dimension(ds_x):
    
    return(np.reshape(ds_x, (ds_x.shape[0], 1, ds_x.shape[1])))

class lenet():
    
    def __init__(self):
        
        self.ds_x_smooth = 0

        
    def prepare_x(self, x_df, augmentRatio=10, smooth = False):
        
        self.x_df = x_df
        
        self.smooth = smooth
        
        self.augmentRatio = augmentRatio
        
        # make scaled absorbance dataset for training
        self.scaler_x = MinMaxScaler()
        self.ds_x = self.x_df.values
        # pca = PCA(n_components = 10)
        # ds_x = pca.fit_transform(ds_x)
        self.scaler_x = self.scaler_x.fit(self.ds_x)

        self.ds_x = self.scaler_x.transform(self.ds_x)
        
        """## Smoothing
        Using Savitzky-Golay
        """

        if self.smooth == True:
        
            window = 5
            degree = 3
            
            self.x = np.array(list(map(lambda s: float(s[4:].replace('_', '.')), self.df.columns[6:])))
            
            self.ds_x_smooth = np.array(list(map(lambda y: savgol_filter((self.x, y),
                                                                         window, degree)[1], self.ds_x)))
             
        else:
            
            self.ds_x_smooth=self.ds_x

        """"DROPPING OUT EVERY N'th WAVELENGTH"""
        self.n_drop = 4
        self.ds_x_smooth = self.ds_x_smooth[:,range(0,self.ds_x_smooth.shape[1],self.n_drop)]
        
        # final step to prepare for using in Sequential() model
        
        self.X_train = self.ds_x_smooth
        
        self.X_train_aug = self.X_train.repeat(self.augmentRatio, axis=0)
        self.X_train_aug = augment(self.X_train_aug, .01, .01, .01)
        
        self.X_train = add_dimension(self.X_train)
        self.X_train_aug = add_dimension(self.X_train_aug)
        
        return(self.X_train)
        
    
    def transform_x(self,x_new_df):
        
        if type(self.ds_x_smooth) == int:
            
            print('ERROR: x must be prepared first by using prepare_x method.')
        
        self.x_new_df = x_new_df

        self.x_new_df = self.scaler_x.transform(self.x_new_df)
        
        """## Smoothing
        Using Savitzky-Golay
        """

        if self.smooth == True:
        
            window = 5
            degree = 3
            
            self.x_new_df = np.array(list(map(lambda y: savgol_filter((self.x, y),
                                                                         window, degree)[1], self.x_new_df)))

        """"DROPPING OUT EVERY N'th WAVELENGTH"""
        self.x_new_df = self.x_new_df[:,range(0,self.x_new_df.shape[1],self.n_drop)]
        
        # final step to prepare for using in Sequential() model
        
        self.x_new_df = add_dimension(self.x_new_df)
        
        return(self.x_new_df)
        
    # def plot_x(self):

    #     plt.figure(figsize=(12, 8))
    #     for i in range(self.ds_x_smooth.shape[0]):
    #         plt.plot(self.ds_x_smooth[i])
    #     plt.xlabel('Wavelength (nm)')
    #     plt.ylabel('Absorbanced (Normalized)')
    #     plt.show()
            
        
    # def smooth_plot(self):
        
    #     if self.smooth == True:
    
    #         plt.figure(figsize=(12, 8))
    #         plt.plot(self.x, self.ds_x[2])
    #         plt.plot(self.x, self.ds_x_smooth[2])
    #         plt.legend(['Original', 'Smooth'])
    #         plt.xlabel('Wavelength (nm)')
    #         plt.ylabel('Absorbance (Normalized)')
    #         plt.title(f'Savitzky-Golay Smoothing (Window Size={window}, Degree={degree})')
    #         plt.show()
            
    #     else:
            
    #         print('x was not smoothed.')
            
        
    def make_model(self, y_train, y_val, X_val, num_epochs = 1000):
        
        """## Augmentation"""
        
        if type(self.ds_x_smooth) == int:
            
            print('ERROR: x must be prepared first by using prepare_x method.')
        
        self.y_train = y_train
        
        self.y_val = y_val
        
        self.X_val = X_val
        
        self.num_epochs = num_epochs

        self.multiplier = self.augmentRatio
    
        self.y_train = self.y_train.to_numpy()
        self.y_train.shape=(len(self.y_train),1)
        
        self.y_train_aug = self.y_train.repeat(self.multiplier, axis=0)
        
        
        
        """# Training Creation
        
        ## LeNet 5
        """
        
        drop = 0.05
        reg = .01
        
        self.model = Sequential()
        self.model.add(Conv1D(6, 1,activation='selu',input_shape=(1, self.X_train_aug.shape[2])))
        self.model.add(Dropout(drop))
        self.model.add(AveragePooling1D(1))
        self.model.add(Conv1D(16, 1, activation='selu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))
        self.model.add(Dropout(drop))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='selu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))
        self.model.add(Dropout(drop))
        self.model.add(Dense(84, activation='selu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))
        self.model.add(Dropout(drop))
        self.model.add(Dense(1,activation = 'linear'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(.001))
        self.history = self.model.fit(
            self.X_train_aug, self.y_train_aug,
            epochs=self.num_epochs, batch_size=int(1e10),
            verbose=0,
            validation_data=(self.X_val, self.y_val),
        )
        
        print('Model trained.')
        
    def predict(self, x_pred):
        
        return(self.model.predict(x_pred).flatten())

#%% Set paths and bring in models

user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
output_dir=os.path.join(path_to_wqs,'Hydroponics/outputs/')
inter_dir=os.path.join(path_to_wqs,'Hydroponics/intermediates/')

abs_wq_df_fn = 'abs-wq_HNSr_df.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)

filename = 'HNSr_DL_Nitrate-N_It0.joblib'
pickle_path = os.path.join(output_dir,'picklejar',filename)

#bring in model
clf = load(pickle_path)

#%% Test model

input_df = abs_wq_df
s = 'Nitrate-N'
iteration = 0

Y = input_df[s]
keep = pd.notna(Y)
X = input_df.loc[keep,'band_1':'band_1024']
Y = Y[keep]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=iteration,
                                                                test_size = 0.3)

X_train = clf.transform_x(X_train)
X_test = clf.transform_x(X_test)

mod = clf

y_hat = mod.predict(X_test)

plt.figure()
plt.scatter(y_test,y_hat)
plt.ylabel('Predicted')
plt.xlabel('True')
