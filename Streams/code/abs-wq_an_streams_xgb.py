# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:39:55 2021

@author: jbarrett.carter
"""
#%% import libraries

import pandas as pd
import numpy as np
import os
import datetime as dt
# import matplotlib.pyplot as plt
# import scipy
from scipy import stats
# import seaborn as sns
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.utils import resample
from sklearn.metrics import mean_squared_error as MSE
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#for looking up available scorers
# import sklearn.metrics
# sorted(sklearn.metrics.SCORERS.keys())

from sklearn.metrics import r2_score

from joblib import dump

from sklearn.base import BaseEstimator

#%% Set paths and bring in data

# user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
# path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
# path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\water_quality-spectroscopy' #for laptop (new)
path_to_wqs = '/blue/ezbean/jbarrett.carter/water_quality-spectroscopy' # for HiPerGator
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')

abs_wq_df_fn = 'abs_wq_df_streams.csv'
# syn_abs_wq_df_fn = 'abs-wq_SWs_OO.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)
# syn_abs_wq_df=pd.read_csv(inter_dir+syn_abs_wq_df_fn)
samp_sizes = pd.read_csv(os.path.join(inter_dir,'fil_sub_samp_sizes.csv'))

#%% seperate into filtered and unfiltered sample sets

abs_wq_df_fil = abs_wq_df.loc[abs_wq_df['Filtered']==True,:]
abs_wq_df_unf = abs_wq_df.loc[abs_wq_df['Filtered']==False,:]

input_df = abs_wq_df_fil # for testing

species = input_df.columns[0:8]
# s = species[2] # for testing

#%% make custom estimator combining PCA and XGB

class pca_xgb(BaseEstimator):
    
    def __init__(self, max_depth=None, learning_rate=None, 
                 n_components=None, detect_lim=None,
                 random_state=None):
        
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_components = n_components
        self.random_state = random_state
        self.detect_lim = detect_lim
        
    def fit(self, X, y):
        
        self.pca=PCA(n_components=self.n_components,
                     random_state=self.random_state)
        self.XGBR = xgb.XGBRegressor(n_estimators = 100,
                                     booster = 'gbtree',
                                     tree_method = 'exact',
                                     random_state=self.random_state,
                                     learning_rate=self.learning_rate,
                                     max_depth=self.max_depth)
        
        keep = y>self.detect_lim
        
        y = y[keep]
        
        X = X.loc[keep,:]
    
        self.pca_fitted = self.pca.fit(X)
        
        X = pd.DataFrame(self.pca.fit_transform(X))
        
        self.scaler_fitted = MinMaxScaler().fit(X)
        
        X = self.scaler_fitted.transform(X)
        
        self.XGBR_fitted=self.XGBR.fit(X,y)
        
        return self
    
    
    def predict(self, X):
        
        X = pd.DataFrame(self.pca_fitted.transform(X))
        X = self.scaler_fitted.transform(X)
        self.y_hat = pd.Series(self.XGBR_fitted.predict(X))
        self.y_hat[self.y_hat<self.detect_lim]=self.detect_lim
        
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
                             
#%% Create function for writing outputs

def create_outputs(input_df,iterations = 1, autosave = False, output_path = None,
                   subset_name = None, syn_aug = False, syn_df = None):
    
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
    
    outputs_df = pd.DataFrame(columns= ['output','species','iteration','value']) #save outputs in dataframe
    
    output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
                    'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
                    'train_rmse','test_mape','train_mape','max_depth','learning_rate',
                    'n_comp']
    
    variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                      'list(X_test.index)','list(X_train.index)','r_sq','r_sq_train','RMSE_test',
                      'RMSE_train','MAPE_test','MAPE_train','max_depth','learning_rate',
                      'n_comp']
    
       
    iteration = 14 # this is for testing
    
    species = input_df.columns[0:8]
    
    if type(iterations)==int:
        iterations = range(iterations)
    
    for s in species:
        for iteration in iterations:
            print('Analyzing '+s)
            print('Iteration - '+str(iteration))
            
            samp_size = samp_sizes.loc[samp_sizes.Species==s,'Samp_size'].values[0]
            
            # samp_size = samp_sizes['Samp_size'].min()
            
            Y = input_df[s]
            keep = Y>0
            
            inter_df = input_df.loc[keep,:]
            
            if sum(keep)>samp_size:
            
                inter_df = inter_df.sample(n = samp_size, random_state = iteration)
            
            X = inter_df.loc[:,'band_1':'band_1024']
            
            Y = inter_df[s]
            
            X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                                random_state=iteration,
                                                                test_size = 0.3)
            
            # X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, 
            #                                                     random_state=iteration,
            #                                                     test_size = 0.2)
            
            if syn_aug:
                
                syn_samp_size = 46
                
                if syn_df.shape[0]>syn_samp_size:
                
                    syn_df = syn_df.sample(n = syn_samp_size, random_state = iteration)
                    
                X_syn = syn_df.loc[:,'band_1':'band_1024']
                
                Y_syn = syn_df[s]
                
                X_train = pd.concat([X_train,X_syn],ignore_index = True)
                y_train = pd.concat([y_train,Y_syn],ignore_index = True)
            
            mod = pca_xgb(random_state=iteration, detect_lim = -100)
            
            param_grid = {'max_depth':stats.randint(1,6),
                          'learning_rate':stats.uniform(loc=0,scale=0.2),
                          'n_components':stats.randint(10,50)}
            
            clf = RandomizedSearchCV(mod,
                                     param_grid,n_iter = 100,
                                     scoring = 'neg_mean_squared_error',
                                     random_state = iteration)

            clf.fit(X_train,y_train)

            mod_opt = clf.best_estimator_
            learning_rate = float(mod_opt.learning_rate)
            max_depth = float(mod_opt.max_depth)
            n_comp=float(mod_opt.n_components)
            # n_est = mod_opt.n_estimators
            # detect_lim = float(mod_opt.detect_lim)
            Y_hat = list(mod_opt.predict(X_test))
            Y_hat_train = list(mod_opt.predict(X_train))
            
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
            
            for out in range(len(output_names)):
                # print(out)
                sub_df = write_output_df(eval(variable_names[out]), output_names[out], s, iteration)
                outputs_df = pd.concat([outputs_df,sub_df],ignore_index=True)
            
            
            filename = f'XGB-PCA_streams-{subset_name}_{s}_It{iteration}.joblib' # for filtration experiments
            # filename = f'XGB-PCA_streams-{subset_name}_syn-aug-{syn_aug}_{s}_It{iteration}.joblib' # for experiments involving synthetic samples
            pickle_path = os.path.join(output_dir,'picklejar',filename)
            dump(clf,pickle_path)
            
            if autosave == True:
                
                outputs_df.to_csv(output_path,index=False)
                  
    return(outputs_df)

#%% Define function for making plots

# def make_plots(outputs_df, output_label):

#     ## make plots for both filtered and unfiltered samples
        
#     fig, axs = plt.subplots(3,3)
#     fig.set_size_inches(15,15)
#     fig.suptitle(output_label,fontsize = 18)
#     fig.tight_layout(pad = 4)
#     axs[2, 2].axis('off')
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
#         x_text = max(line11)-(max(line11)-min(line11))*0.5
        
#         train_rmse = outputs_df['value'][(outputs_df.output == 'train_rmse')&
#                             (outputs_df.species==s)]
        
#         train_rmse = np.mean(train_rmse)
        
#         test_rmse = outputs_df['value'][(outputs_df.output == 'test_rmse')&
#                             (outputs_df.species==s)]
        
#         test_rmse = np.mean(test_rmse)
        
#         learning_rate = outputs_df['value'][(outputs_df.output == 'learning_rate')&
#                             (outputs_df.species==s)]
        
#         learning_rate = np.mean(learning_rate)
        
#         max_depth = outputs_df['value'][(outputs_df.output == 'max_depth')&
#                             (outputs_df.species==s)]
        
#         ax = axs[row,col]
        
#         for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#             label.set_fontsize(16)
        
#         axs[row,col].plot(y_true_train,y_hat_train,'o',markersize = 4, label = 'training set')
#         axs[row,col].plot(y_true_test,y_hat_test,'o',markersize = 4, label = 'test set')
#         axs[row,col].plot(line11,line11,'k--',label= '1:1 line')
#         # axs[row,col].set_title(s)
#         if (row == 0) and (col == 0):
#             axs[row,col].legend(loc = 'upper left',fontsize = 16)
#         axs[row,col].set_xlabel('Lab Measured '+s+' (mg/L)',fontsize = 16)
#         axs[row,col].set_ylabel('Predicted '+s+' (mg/L)',fontsize = 16)
#         # axs[row,col].get_xaxis().set_visible(False)
#         # ax.text(x_text,y_text,'$rmse_{tr} =$'+str(np.round(train_rmse,3))+'\n'
#         #         +'$rmse_{te} =$'+str(np.round(test_rmse,3))+'\n'
#         #         +'$lr =$'+str(np.round(learning_rate,2))+'\n'
#         #         +'$md =$'+str(int(max_depth)), fontsize = 16)
#         ax.text(x_text,y_text,'$rmse_{tr} =$'+str(np.round(train_rmse,3))+'\n'
#                 +'$rmse_{te} =$'+str(np.round(test_rmse,3)), fontsize = 16)
#         # ticks = ax.get_yticks()
#         # print(ticks)
#         # # tick_labels = ax.get_yticklabels()
#         # tick_labels =[str(round(x,1)) for x in ticks]
#         # tick_labels = tick_labels[1:-1]
#         # print(tick_labels)
#         # ax.set_xticks(ticks)
#         # ax.set_xticklabels(tick_labels)
        
#         if col == 2:
#             col = 0
#             row += 1
#         else:
#             col +=1
#     # fig.show()

#%% function to make and save outputs

# def make_and_save_outputs(input_df,output_path,iterations = 1):
#     outputs_df = create_outputs(input_df,iterations)
#     outputs_df.to_csv(output_path,index=False)
#     return(outputs_df)

#%% Create outputs for models trained with filtered, unfiltered, and all samples

# train_start = dt.datetime.now()

# outputs_df = create_outputs(abs_wq_df,iterations = 1) # all samples

# outputs_df = create_outputs(abs_wq_df,iterations = np.linspace(3,19,17,dtype=int)) # all samples

create_outputs(abs_wq_df_fil, iterations = 20, autosave = True,
               output_path = os.path.join(output_dir,'streams-fil_XGB-PCA_It0-19_results.csv'),
               subset_name = 'fil') # filtered samples

create_outputs(abs_wq_df_unf, iterations = 20, autosave = True,
               output_path = os.path.join(output_dir,'streams-unf_XGB-PCA_It0-19_results.csv'),
               subset_name = 'unf') # unfiltered samples

create_outputs(abs_wq_df, iterations = 20, autosave = True,
               output_path = os.path.join(output_dir,'streams-comb_XGB-PCA_It0-19_results.csv'),
               subset_name = 'comb') # all samples

# create_outputs(abs_wq_df_fil, iterations = 20, autosave = True,
#                output_path = os.path.join(output_dir,'streams-fil_syn-aug-False_XGB-PCA_It0-19_results.csv'),
#                subset_name = 'fil',syn_aug = False) # filtered samples, no synthetic samples

# create_outputs(abs_wq_df_fil, iterations = 20, autosave = True,
#                output_path = os.path.join(output_dir,'streams-fil_syn-aug-True_XGB-PCA_It0-19_results.csv'),
#                subset_name = 'fil',syn_aug = True, syn_df = syn_abs_wq_df) # filtered samples with synthetic samples

# train_stop = dt.datetime.now()
 
#%% make plots for all samples

# make_plots(outputs_df,'Filtered and Unfiltered Stream Samples - XGB-PCA')

#%% save output

# outputs_df.to_csv(output_dir+'streams_XGB_It0_results.csv',index=False)
   
#%% make and save output.

# outputs_df = make_and_save_outputs(abs_wq_df,output_dir+'streams_XGB-PCA_It0-19_results.csv',
#                        iterations = 20)

#%%
