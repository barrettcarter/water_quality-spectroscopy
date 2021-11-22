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
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE

#%%
################################################################################ DEFAULTS

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
spectra_path = os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir = os.path.join(path_to_wqs,'Streams/outputs/')
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

#%% Creat function for writing output files

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

#%% Create function for producing and writing model outputs

def create_outputs(input_df,num_epochs = 1000,iterations = 1):
    
    ### Create a model for every species
    # s = 'Molybdenum'
    
    outputs_df = pd.DataFrame(columns= ['output','species','iteration','value']) #save outputs in dataframe
    
    output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
                    'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
                    'train_rmse','test_mape','train_mape']
    
    variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                      'y_test_ind','y_train_ind','r_sq','r_sq_train','RMSE_test',
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
        
        for iteration in range(iterations):
            
            X_train=ds_x_smooth[df[s]>0,:]

            y_train = df.loc[df[s]>0,s]
            
            # X_train = pd.DataFrame(X_train)
            
            """## Train/Val Split"""
            print('Full set:',X_train.shape)
            
            X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=.25,
                                                              random_state= iteration)
            X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=.20,
                                                                  random_state=iteration)
            
            y_train_ind = list(y_train.index)
            y_test_ind = list(y_test.index)
            
            """## Augmentation"""
            
            multiplier = augmentRatio

            y_train = y_train.to_numpy()
            y_train.shape=(len(y_train),1)
            
            y_train = y_train.repeat(multiplier, axis=0)
            
            X_train = X_train.repeat(multiplier, axis=0)
            X_train = augment(X_train, .01, .01, .01)
            
            print(s)
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
            lenet.add(Dense(1,activation = 'linear'))
            lenet.compile(loss='mean_squared_error', optimizer=Adam(.001))
            history = lenet.fit(
                X_train, y_train,
                epochs=num_epochs, batch_size=int(1e10),
                verbose=0,
                validation_data=(X_val, y_val),
            )
        
            Y_hat = list(predict([lenet], X_test))
            Y_hat_train = list(predict([lenet], X_train))
            
            r_sq = float(r2_score(y_test,Y_hat))
            r_sq_train = float(r2_score(y_train,Y_hat_train))
            
            print('Training r-squared: '+str(r_sq_train))
            print('Test r-squared: '+str(r_sq))
            
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
            
            # val_i_y = predict([lenet], X_test)
            # val_v_y = predict([lenet], X_val)
            # val_m_y = predict([lenet], X_train)
            
            # mod_x, mod_y = (np.expand_dims(y_train.flatten(), axis=1),np.expand_dims(val_m_y.flatten(), axis=1))
            # regr = linear_model.LinearRegression()
            # regr.fit(mod_x, mod_y)
            # y_pred = regr.predict(mod_x)
            # print('Model calibration',r2_score(mod_y, y_pred))
            
            # int_x, int_y = (np.expand_dims(y_val.flatten(), axis=1),np.expand_dims(val_v_y.flatten(), axis=1))
            # regr = linear_model.LinearRegression()
            # regr.fit(int_x, int_y)
            # y_pred = regr.predict(int_x)
            # print('Internal validation',r2_score(int_y, y_pred))
            
            # ext_x, ext_y = (np.expand_dims(y_test.flatten(), axis=1),np.expand_dims(val_i_y.flatten(), axis=1))
            # regr = linear_model.LinearRegression()
            # regr.fit(ext_x, ext_y)
            # y_pred = regr.predict(ext_x)
            
            # print('External validation',r2_score(ext_x, ext_y))
            # plt.figure(figsize=(12,8))
            # plt.scatter(ext_x, ext_y, c='red',alpha=0.5,s=200)
            # plt.scatter(int_x, int_y, c='blue',alpha=0.5,s=200)
            # plt.scatter(mod_x, mod_y, c='none',alpha=0.5,s=200,edgecolors='black')
            # #plt.plot(x, y_pred, linewidth=3)
            # plt.plot([np.min(mod_x),np.max(mod_x)], [np.min(mod_x),np.max(mod_x)], color='k', linestyle='--', linewidth=2)
            
            # #plt.ylim(0, 1)
            # #plt.xlim(0, 1)
            # plt.xlabel('Observed',fontsize=14)
            # plt.ylabel('Predicted',fontsize=14)
            # plt.title(s+f' model (R^2={round(r2_score(ext_x, ext_y), 2)})',fontsize=14)
            # plt.show()
        
    return(outputs_df)

#%% Create outputs

outputs_df = create_outputs(abs_wq_df,num_epochs=5000,iterations = 1)

#%% Define function for making plots

def make_plots(outputs_df, output_label):

    ## make plots for both filtered and unfiltered samples
        
    fig, axs = plt.subplots(3,2)
    fig.set_size_inches(10,15)
    fig.suptitle(output_label,fontsize = 18)
    fig.tight_layout(pad = 4)
    axs[2, 1].axis('off')
    row = 0
    col = 0
    species = outputs_df.species.unique()
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
        
        y_text = min(line11)+(max(line11)-min(line11))*0
        x_text = max(line11)-(max(line11)-min(line11))*0.5
        
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
        
        train_rsq = float(outputs_df['value'][(outputs_df.output == 'train_rsq')&
                            (outputs_df.species==s)])
        
        test_rsq = float(outputs_df['value'][(outputs_df.output == 'test_rsq')&
                            (outputs_df.species==s)])
        
        ax = axs[row,col]
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)
        
        axs[row,col].plot(y_true_train,y_hat_train,'o',markersize = 4, label = 'training set')
        axs[row,col].plot(y_true_test,y_hat_test,'o',markersize = 4, label = 'test set')
        axs[row,col].plot(line11,line11,'k--',label= '1:1 line')
        # axs[row,col].set_title(s)
        axs[row,col].legend(loc = 'upper left',fontsize = 16)
        axs[row,col].set_xlabel('Lab Measured '+s+' (mg/L)',fontsize = 16)
        axs[row,col].set_ylabel('Predicted '+s+' (mg/L)',fontsize = 16)
        # axs[row,col].get_xaxis().set_visible(False)
        ax.text(x_text,y_text,r'$train\/r^2 =$'+str(np.round(train_rsq,3))+'\n'
                +r'$test\/r^2 =$'+str(np.round(test_rsq,3)), fontsize = 16)
        # ticks = ax.get_yticks()
        # print(ticks)
        # # tick_labels = ax.get_yticklabels()
        # tick_labels =[str(round(x,1)) for x in ticks]
        # tick_labels = tick_labels[1:-1]
        # print(tick_labels)
        # ax.set_xticks(ticks)
        # ax.set_xticklabels(tick_labels)
        
        if col == 1:
            col = 0
            row += 1
        else:
            col +=1
    # fig.show()
    
#%% make plots for all samples

make_plots(outputs_df,'Filtered and Unfiltered Samples')
# make_plots(outputs_df_fil,'Filtered Samples')
# make_plots(outputs_df_unf,'Unfiltered Samples')

#%% Save output file

outputs_df.to_csv(output_dir+'streams_DL_B1_results.csv',index=False)

#%% make and save outputs

def make_and_save_outputs(input_df,output_path,its = 1,eps = 1000):
    outputs_df = create_outputs(input_df,iterations = its, num_epochs = eps)
    outputs_df.to_csv(output_path,index=False)
    
#%% do it.

make_and_save_outputs(abs_wq_df,output_dir+'streams_DL_B10_results.csv',
                      its = 10, eps = 5000)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Testing

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
            
#%%

v = abs_wq_df['TKN']
eval('v.index')

#%%

v = v.values
v = v.repeat(2,axis = 0)

