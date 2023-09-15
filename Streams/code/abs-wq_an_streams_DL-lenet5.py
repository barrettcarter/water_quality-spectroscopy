# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:14:36 2021

@author: aditya01
"""

print('loading modules...')

import os
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
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
# from sklearn.decomposition import PCA

from joblib import dump

print('modules loaded')

#%% Set paths and bring in data

# user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
path_to_wqs = '/blue/ezbean/jbarrett.carter/water_quality-spectroscopy/' # for HiPerGator
# path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
int_dir = os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir = os.path.join(path_to_wqs,'Streams/outputs/')
abs_wq_fn = 'abs_wq_df_streams.csv'
# syn_abs_wq_df_fn = 'abs-wq_SWs_OO.csv'
spectra_path = os.path.join(int_dir,abs_wq_fn)
os.path.exists(spectra_path)
np.random.seed(7)

samp_sizes = pd.read_csv(os.path.join(int_dir,'fil_sub_samp_sizes.csv'))

#%% seperate into filtered and unfiltered sample sets

species=['Ammonium-N','Nitrate-N','TKN','ON','TN','Phosphate-P','TP','OP']

# species=['Nitrate-N'] # for testing

# s = species[0]

abs_wq_df=pd.read_csv(spectra_path)

# syn_abs_wq_df=pd.read_csv(int_dir+syn_abs_wq_df_fn)

abs_wq_df_fil = abs_wq_df.loc[abs_wq_df['Filtered']==True,:]
abs_wq_df_unf = abs_wq_df.loc[abs_wq_df['Filtered']==False,:]

specCols=[x for x in abs_wq_df.columns if x.startswith('band_')]

# abs_wq_df=abs_wq_df.dropna(axis=0)

# ds_x=df.loc[:,specCols].values

# abs_wq_df.loc[abs_wq_df['Phosphate-P']<0.15,['Phosphate-P','TP','OP']]=-0.1

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

#%% Create a function for making the outputs

def make_outputs(df,num_epochs,outputs_df,s,iteration,output_names,
                 variable_names, output_path = None, autosave = False,
                 subset_name = None, syn_aug = False, syn_df = None):

    print(f'working on species: {s}')
    print(f'iteration {iteration}')
    
    samp_size = samp_sizes.loc[samp_sizes.Species==s,'Samp_size'].values[0] # for filtration experiments
    
    # samp_size = samp_sizes['Samp_size'].min() # for synthetic sample experiments
    
    keep = df[s]>0
    
    df = df.loc[keep,:]
    
    if sum(keep)>samp_size:
    
        df = df.sample(n = samp_size, random_state = iteration)
    
    X_train=df.loc[:,specCols]

    y_train = df[s]
    
    # X_train = pd.DataFrame(X_train)
    
    """## Train/Val Split"""
    print('Full set:',X_train.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.3,
                                                      random_state= iteration)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.20,
                                                          random_state=iteration)
    
    if syn_aug:
        
        syn_samp_size = 46
        
        if syn_df.shape[0]>syn_samp_size:
        
            syn_df = syn_df.sample(n = syn_samp_size, random_state = iteration)
            
        X_syn = syn_df.loc[:,'band_1':'band_1024']
        
        Y_syn = syn_df[s]
        
        X_train = pd.concat([X_train,X_syn],ignore_index = True)
        y_train = pd.concat([y_train,Y_syn],ignore_index = True)
    
    print(s)
    print('Train set:',X_train.shape)
    print('Validation set:',X_val.shape)
    print('External set:',X_test.shape)
    
    # initialize lenet model and prepare X sets for modeling
    
    lenet_mod = lenet()
    
    X_train = lenet_mod.prepare_x(X_train)
    # print(f' X_train shape: {X_train.shape}')
    
    
    X_val = lenet_mod.transform_x(X_val)
    X_test = lenet_mod.transform_x(X_test)
    # print(f' X_test shape: {X_test.shape}')
    
    lenet_mod.make_model(y_train,y_val,X_val,num_epochs = num_epochs)
    
    print(f'y_train shape: {y_train.shape}')
    print(f'y_train type: {type(y_train)}')
    
    y_train_ind = list(y_train.index)
    y_test_ind = list(y_test.index)
    
    y_train = list(y_train)
    
    Y_hat = list(lenet_mod.predict(X_test))
    # print(f' Y_hat: {Y_hat}')
    Y_hat_train = list(lenet_mod.predict(X_train))
    # print(f' Y_hat_train: {Y_hat_train}')
    
    r_sq = float(r2_score(y_test,Y_hat))
    r_sq_train = float(r2_score(y_train,Y_hat_train))
    
    print('Training r-squared: '+str(r_sq_train))
    print('Test r-squared: '+str(r_sq))
    
    MSE_test = MSE(y_test,Y_hat)
    RMSE_test = float(np.sqrt(MSE_test))
    
    MSE_train = MSE(y_train,Y_hat_train)
    RMSE_train = float(np.sqrt(MSE_train))
    
    # abs_test_errors = abs(y_test-Y_hat)
    # APE_test = abs_test_errors/y_test # APE = absolute percent error,decimal
    # MAPE_test = float(np.mean(APE_test)*100) # this is percentage
    
    # abs_train_errors = abs(y_train-Y_hat_train)
    # APE_train = abs_train_errors/y_train # APE = absolute percent error,decimal
    # MAPE_train = float(np.mean(APE_train)*100) # this is percentage
    
    
    filename = f'DL_streams-{subset_name}_{s}_It{iteration}.joblib' # for filtration experiments
    # filename = f'DL_streams-{subset_name}_syn-aug-{syn_aug}_{s}_It{iteration}.joblib' # for synthetic sample experiments
    pickle_path = os.path.join(output_dir,'picklejar',filename)
    dump(lenet_mod,pickle_path)
    
    ### Write outputs
    
    for out in range(len(output_names)):
        # print(type(out))
        try:
            print(f'variable name: {variable_names[out]}')
            print(f'output name: {output_names[out]}')
            print(f'variable type: {type(eval(variable_names[out]))}')
            # print(type(eval(output_names[out])))
            sub_df = write_output_df(eval(variable_names[out]), output_names[out], s, iteration)
            outputs_df = pd.concat([outputs_df,sub_df],ignore_index=True)
        except AttributeError as e:
            print(e)
            import sys
            sys.exit(0)
            
    if autosave == True:
        
        outputs_df.to_csv(output_path,index=False)
            
    return(outputs_df)
    
    

################################################################################ CHECK FITS

    # plt.figure(figsize=(12,8))
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # #plt.ylim((0, .05))
    # # plt.ylim(0, 0.05)
    # plt.legend(['Training', 'Validation'], loc='upper right')
    # plt.show()
    
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

#%% Create function for producing and writing model outputs
"""
This function takes in an input pandas data frame 'input_df', which has a column
for each species' concentrations and a column for absorbance at each wavelength.

'num_epochs' is an integer used in the artificial neural network algorithm.

'iterations' specifies the values to be used as the random seed for the
resampling procedure. The length of 'iterations' determines the number of times
resampling is performed. 'iterations' can be an integer, float, string, range,
or 1-D numpy array.

"""
def create_outputs(input_df,num_epochs = 5000,iterations = 1, autosave = False,
                   output_path = None, subset_name = None, syn_aug = False, syn_df = None):
    
    
    ### Create a model for every species
    # s = 'Molybdenum'
    
    outputs_df = pd.DataFrame(columns= ['output','species','iteration','value']) #save outputs in dataframe
    
    # output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
    #                 'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
    #                 'train_rmse','test_mape','train_mape']
    
    # variable_names = ['Y_hat','Y_hat_train','list(y_train[:,0])', 'list(y_test)',
    #                   'y_test_ind','y_train_ind','r_sq','r_sq_train','RMSE_test',
    #                   'RMSE_train','MAPE_test','MAPE_train']

    output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
                'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
                'train_rmse']
    
    variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                      'y_test_ind','y_train_ind','r_sq','r_sq_train','RMSE_test',
                      'RMSE_train']
################################################################################ PROCESS

    df = input_df
    for s in species:
        
        if type(iterations)==int:
            
            iterations = range(iterations)
        
        for iteration in iterations:
            
            outputs_df = make_outputs(df,num_epochs,outputs_df,s,iteration,
                         output_names,variable_names, output_path = output_path,
                         autosave = autosave, subset_name = subset_name,
                         syn_aug = syn_aug, syn_df = syn_df)
        
    # return(outputs_df)

#%% Create outputs

# outputs_df = create_outputs(abs_wq_df,num_epochs=1000,iterations = 0)

create_outputs(abs_wq_df_fil, iterations = range(20), autosave = True,
               output_path = os.path.join(output_dir,'streams-fil_DL_It0-19_results.csv'),
               subset_name = 'fil') # filtered samples

create_outputs(abs_wq_df_unf, iterations = range(20), autosave = True,
                output_path = os.path.join(output_dir,'streams-unf_DL_It0-19_results.csv'),
                subset_name = 'unf') # unfiltered samples

create_outputs(abs_wq_df, iterations = range(20), autosave = True,
                output_path = os.path.join(output_dir,'streams-comb_DL_It0-19_results.csv'),
                subset_name = 'comb') # all samples

# create_outputs(abs_wq_df_fil, iterations = 20, autosave = True,
#                output_path = os.path.join(output_dir,'streams-fil_syn-aug-False_DL_It0-19_results.csv'),
#                subset_name = 'fil',syn_aug = False) # filtered samples, no synthetic samples

# create_outputs(abs_wq_df_fil, iterations = 20, autosave = True,
#                output_path = os.path.join(output_dir,'streams-fil_syn-aug-True_DL_It0-19_results.csv'),
#                subset_name = 'fil',syn_aug = True, syn_df = syn_abs_wq_df) # filtered samples with synthetic samples

#%% Define function for making plots

# def make_plots(outputs_df, output_label):

#     ## make plots for both filtered and unfiltered samples
        
#     fig, axs = plt.subplots(4,2)
#     fig.set_size_inches(15,20)
#     fig.suptitle(output_label,fontsize = 18)
#     fig.tight_layout(pad = 4)
#     #axs[2, 1].axis('off')
#     row = 0
#     col = 0
#     species = outputs_df.species.unique()
#     for s in species:
#         y_true_train = outputs_df.loc[((outputs_df.species == s) &
#                                         (outputs_df.output == 'y_true_train')),
#                                        'value']
        
#         y_hat_train = outputs_df.loc[((outputs_df.species == s) &
#                                         (outputs_df.output == 'y_hat_train')),
#                                        'value']
        
#         y_true_test = outputs_df.loc[((outputs_df.species == s) &
#                                         (outputs_df.output == 'y_true_test')),
#                                        'value']
        
#         y_hat_test = outputs_df.loc[((outputs_df.species == s) &
#                                         (outputs_df.output == 'y_hat_test')),
#                                        'value']
        
#         line11 = np.linspace(min(np.concatenate((y_true_train,y_hat_train,
#                                                  y_true_test,y_hat_test))),
#                               max(np.concatenate((y_true_train,y_hat_train,
#                                                  y_true_test,y_hat_test))))
        
#         y_text = min(line11)+(max(line11)-min(line11))*0
#         x_text = max(line11)-(max(line11)-min(line11))*0.6
        
#         # lr = LinearRegression().fit(Y_hat,y_test)
#         # linelr = lr.predict(line11.reshape(-1,1))
        
#         # plt.plot(y_true_train,y_hat_train,'o',markersize = 4, label = 'predictions')
#         # plt.plot(line11,line11,label= '1:1 line')
#         # plt.title(s)
#         # # plt.plot(line11,linelr,label = 'regression line')
#         # plt.xlabel('Lab Measured '+s+' (mg/L)')
#         # plt.ylabel('Predicted '+s+' (mg/L)')
#         # plt.text(x_text,y_text1,r'$r^2 =$'+str(np.round(r_sq,3)))
#         # plt.text(x_text,y_text2,r'MAPE = '+str(np.round(MAPE_test,1))+'%')
#         # plt.legend()
#         # plt.show()
        
#         train_rmse = float(outputs_df['value'][(outputs_df.output == 'train_rmse')&
#                             (outputs_df.species==s)])
        
#         test_rmse = float(outputs_df['value'][(outputs_df.output == 'test_rmse')&
#                             (outputs_df.species==s)])
        
#         ax = axs[row,col]
        
#         for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#             label.set_fontsize(16)
        
#         axs[row,col].plot(y_true_train,y_hat_train,'o',markersize = 4, label = 'training set')
#         axs[row,col].plot(y_true_test,y_hat_test,'o',markersize = 4, label = 'test set')
#         axs[row,col].plot(line11,line11,'k--',label= '1:1 line')
#         # axs[row,col].set_title(s)
#         axs[row,col].legend(loc = 'upper left',fontsize = 16)
#         axs[row,col].set_xlabel('Lab Measured '+s+' (mg/L)',fontsize = 16)
#         axs[row,col].set_ylabel('Predicted '+s+' (mg/L)',fontsize = 16)
#         # axs[row,col].get_xaxis().set_visible(False)
#         ax.text(x_text,y_text,r'$train\/rmse =$'+str(np.round(train_rmse,3))+'\n'
#                 +r'$test\/rmse =$'+str(np.round(test_rmse,3)), fontsize = 16)
#         # ticks = ax.get_yticks()
#         # print(ticks)
#         # # tick_labels = ax.get_yticklabels()
#         # tick_labels =[str(round(x,1)) for x in ticks]
#         # tick_labels = tick_labels[1:-1]
#         # print(tick_labels)
#         # ax.set_xticks(ticks)
#         # ax.set_xticklabels(tick_labels)
        
#         if col == 1:
#             col = 0
#             row += 1
#         else:
#             col +=1
#     # fig.show()
    
#%% make plots for all samples

# make_plots(outputs_df,'Filtered and Unfiltered Samples')
# # make_plots(outputs_df_fil,'Filtered Samples')
# # make_plots(outputs_df_unf,'Unfiltered Samples')

#%% Save output file

# outputs_df.to_csv(output_dir+'streams_DL_B1_results.csv',index=False)

#%% make and save outputs

# def make_and_save_outputs(input_df,output_path,its = 1,eps = 1000):
#     outputs_df = create_outputs(input_df,iterations = its, num_epochs = eps)
#     outputs_df.to_csv(output_path,index=False)
    
#%% do it.

# make_and_save_outputs(abs_wq_df,output_dir+'streams_DL_It0-19_results.csv',
#                       its = np.arange(0,20), eps = 5000)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# # Testing

# output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
#                     'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
#                     'train_rmse','test_mape','train_mape']
    
# variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
#                       'list(X_test.index)','list(X_train.index)','r_sq','r_sq_train','RMSE_test',
#                       'RMSE_train','MAPE_test','MAPE_train']

# s = 'something'
# iteration = 11041991

# for out in range(len(output_names)):
#             # print(out)
#             print(eval(variable_names[out])) 
#             print(output_names[out]) 
#             print(s)
#             print(iteration)
            
# #%%

# v = abs_wq_df['TKN']
# eval('v.index')

# #%%

# v = v.values
# v = v.repeat(2,axis = 0)

