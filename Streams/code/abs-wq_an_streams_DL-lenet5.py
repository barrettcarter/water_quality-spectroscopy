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
from sklearn.metrics import mean_squared_error as MSE

#%%
################################################################################ DEFAULTS

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
spectra_path = os.path.join(path_to_wqs,'Streams/intermediates/')
abs_wq_fn = 'abs_wq_df_streams.csv'
spectra_path = os.path.join(spectra_path,abs_wq_fn)
os.path.exists(spectra_path)
np.random.seed(7)

#%%

species=['Nitrate-N', 'Ammonium-N', 'Phosphate-P','TKN','TP']

iEpochs=5000

augmentRatio=7

abs_wq_df=pd.read_csv(spectra_path)

specCols=[x for x in abs_wq_df.columns if x.startswith('band_')]

abs_wq_df=abs_wq_df.dropna(axis=0)

# ds_x=df.loc[:,specCols].values


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

#%% Create function for producing and writing model outputs

def create_outputs(input_df,num_epochs = 1000):
    
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
    
    ### Create a model for every species
    # s = 'Molybdenum'
    
    outputs_df = pd.DataFrame(columns= ['output','species','iteration','value']) #save outputs in dataframe
    
    output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
                    'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
                    'train_rmse','test_mape','train_mape']
    
    variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                      'list(X_test.index)','list(X_train.index)','r_sq','r_sq_train','RMSE_test',
                      'RMSE_train','MAPE_test','MAPE_train']
################################################################################ PROCESS

    scaler_x = MinMaxScaler()
    ds_x = input_df.loc[:,specCols].values
    ds_x = scaler_x.fit_transform(ds_x)
    # scaler_y = MinMaxScaler()
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
    
    # v = species[12]
    # v="Molybdenum"
    iteration = 1
    df = input_df
    for s in species:
        
        ds_y=df.loc[df[s]>0,s].values
        # ds_y=df[[s]][df[[s]]>0].values
        ds_y.shape=(len(ds_y),1)
        y_train=ds_y
        
        """## Augmentation TEST"""
        
        # x = X_train[0:1]
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
        
        X_train=ds_x_smooth[df[s]>0,:]
        multiplier = augmentRatio
        X_train = X_train.repeat(multiplier, axis=0)
        X_train = augment(X_train, .01, .01, .01)
        
        y_train = y_train.repeat(multiplier, axis=0)
        
        # X_train = pd.DataFrame(X_train)
        
        """## Train/Val Split"""
        print('Full set:',X_train.shape)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=.25,
                                                          random_state= iteration)
        X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=.20,
                                                              random_state=iteration)
        
        print('Train set:',X_train.shape)
        print('Validation set:',X_val.shape)
        print('External set:',X_test.shape)
        
        """## Add Dimension
        """
        X_train = add_dimension(X_train)
        X_val = add_dimension(X_val)
        X_test = add_dimension(X_test)
        
        """# Training Creation
        
        ## LeNet 5
        """
        
        drop = 0.05
        reg = .01
        
        lenet = Sequential()
        lenet.add(Conv1D(6, 1,activation='selu',input_shape=(1, X_train.shape[2])))
        lenet.add(Dropout(drop))
        lenet.add(AveragePooling1D(1))
        lenet.add(Conv1D(16, 1, activation='selu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))
        lenet.add(Dropout(drop))
        lenet.add(Flatten())
        lenet.add(Dense(120, activation='selu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))
        lenet.add(Dropout(drop))
        lenet.add(Dense(84, activation='selu', kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))
        lenet.add(Dropout(drop))
        lenet.add(Dense(y_train.shape[1],activation = 'linear'))
        lenet.compile(loss='mean_squared_error', optimizer=Adam(.001))
        history = lenet.fit(
            X_train, y_train,
            epochs=num_epochs, batch_size=int(1e10),
            verbose=2,
            validation_data=(X_val, y_val),
        )
    
        Y_hat = list(predict([lenet], X_test))
        Y_hat_train = list(predict([lenet], X_train))
        
        r_sq = float(r2_score(y_test,Y_hat))
        r_sq_train = float(r2_score(y_train,Y_hat_train))
        
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
        
        ### Write outputs
        
        for out in range(len(output_names)):
            print(type(out))
            try:
                print(variable_names[out])
                print(type(variable_names[out]))
                print(type(output_names[out]))
                sub_df = write_output_df(eval(variable_names[out]), output_names[out], s, iteration)
                outputs_df = outputs_df.append(sub_df,ignore_index=True)
            except AttributeError as e:
                print(e)
                import sys
                sys.exit(0)
        
        return(outputs_df)
    
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
        
        val_i_y = predict([lenet], X_test)
        val_v_y = predict([lenet], X_val)
        val_m_y = predict([lenet], X_train)
        
        mod_x, mod_y = (np.expand_dims(y_train.flatten(), axis=1),np.expand_dims(val_m_y.flatten(), axis=1))
        regr = linear_model.LinearRegression()
        regr.fit(mod_x, mod_y)
        y_pred = regr.predict(mod_x)
        print('Model calibration',r2_score(mod_y, y_pred))
        
        int_x, int_y = (np.expand_dims(y_val.flatten(), axis=1),np.expand_dims(val_v_y.flatten(), axis=1))
        regr = linear_model.LinearRegression()
        regr.fit(int_x, int_y)
        y_pred = regr.predict(int_x)
        print('Internal validation',r2_score(int_y, y_pred))
        
        ext_x, ext_y = (np.expand_dims(y_test.flatten(), axis=1),np.expand_dims(val_i_y.flatten(), axis=1))
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
        plt.title(s+f' model (R^2={round(r2_score(ext_x, ext_y), 2)})',fontsize=14)
        plt.show()

#%% Create outputs

outputs_df = create_outputs(abs_wq_df,num_epochs=1000)

#%%

output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
                    'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
                    'train_rmse','test_mape','train_mape']
    
variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                      'list(X_test.index)','list(X_train.index)','r_sq','r_sq_train','RMSE_test',
                      'RMSE_train','MAPE_test','MAPE_train']

s = 'something'
iteration = 11041991

for out in range(len(output_names)):
            # print(out)
            print(eval(variable_names[out])) 
            print(output_names[out]) 
            print(s)
            print(iteration)
            
        

