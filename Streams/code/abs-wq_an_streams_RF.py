# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:39:55 2021

@author: jbarrett.carter
"""
#%% import libraries

import pandas as pd
import numpy as np
import os
# import datetime as dt
import matplotlib.pyplot as plt
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
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#for looking up available scorers
import sklearn.metrics
sorted(sklearn.metrics.SCORERS.keys())

from joblib import dump

#%% Set paths and bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')

abs_wq_df_fn = 'abs_wq_df_streams.csv'

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)

#%% seperate into filtered and unfiltered sample sets

abs_wq_df_fil = abs_wq_df.loc[abs_wq_df['Filtered']==True,:]
abs_wq_df_unf = abs_wq_df.loc[abs_wq_df['Filtered']==False,:]
                             
#%% Create function for writing outputs

def create_outputs(input_df,iterations = 1):
    
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
    s = 'Molybdenum'
    
    outputs_df = pd.DataFrame(columns= ['output','species','iteration','value']) #save outputs in dataframe
    
    output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
                    'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
                    'train_rmse','test_mape','train_mape','max_features','ccp_alpha']
    
    variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                      'list(X_test.index)','list(X_train.index)','r_sq','r_sq_train','RMSE_test',
                      'RMSE_train','MAPE_test','MAPE_train','max_features','ccp_alpha']
    
       
    iteration = 1 # this is for testing
    
    species = input_df.columns[0:8]
    
    if type(iterations)==int:
        iterations = range(iterations)
    
    for s in species:
        for iteration in iterations:
            print('Analyzing '+s)
            print('Iteration - '+str(iteration))
            Y = input_df[s]
            keep = pd.notna(Y)
            X = input_df.loc[keep,'band_1':'band_1024']
            
            # dimensional reduction
            # pca = PCA(n_components = 20)
            # X = pd.DataFrame(pca.fit_transform(X))
            # X = MinMaxScaler(X)
            
            Y = Y[keep]
            
            X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=iteration,
                                                                test_size = 0.3)

            param_grid = {'max_features':stats.uniform(scale = 0.01),
                          'ccp_alpha':stats.uniform(scale=0.01)}
            
            clf = RandomizedSearchCV(RF(n_estimators = 1000,random_state=iteration),
                                     param_grid,n_iter = 20,
                                     scoring = 'neg_mean_absolute_error',
                                     random_state = iteration)

            clf.fit(X_train,y_train)
            max_features = float(clf.best_params_['max_features'])
            ccp_alpha = float(clf.best_params_['ccp_alpha'])
            mod_opt = clf.best_estimator_
            Y_hat = list(mod_opt.predict(X_test))
            Y_hat_train = list(mod_opt.predict(X_train))
            
            r_sq = float(mod_opt.score(X_test,y_test))
            r_sq_train = float(mod_opt.score(X_train,y_train))
            
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
                outputs_df = outputs_df.append(sub_df,ignore_index=True)
                
            filename = f'RF_{s}_It{iteration}.joblib'
            pickle_path = os.path.join(output_dir,'picklejar',filename)
            dump(clf,pickle_path)
                  
    return(outputs_df)

#%% Define function for making plots

def make_plots(outputs_df, output_label):

    ## make plots for both filtered and unfiltered samples
        
    fig, axs = plt.subplots(3,3)
    fig.set_size_inches(15,15)
    fig.suptitle(output_label,fontsize = 18)
    fig.tight_layout(pad = 4)
    axs[2, 2].axis('off')
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
        
        train_rsq = outputs_df['value'][(outputs_df.output == 'train_rsq')&
                            (outputs_df.species==s)]
        
        train_rsq = np.mean(train_rsq)
        
        test_rsq = outputs_df['value'][(outputs_df.output == 'test_rsq')&
                            (outputs_df.species==s)]
        
        test_rsq = np.mean(test_rsq)
        
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
        
        if col == 2:
            col = 0
            row += 1
        else:
            col +=1
    # fig.show()

#%% function for make and save outputs

def make_and_save_outputs(input_df,output_path,iterations = 1):
    outputs_df = create_outputs(input_df,iterations)
    outputs_df.to_csv(output_path,index=False)

#%% Create outputs for models trained with filtered, unfiltered, and all samples

outputs_df = create_outputs(abs_wq_df) # all samples
outputs_df_fil = create_outputs(abs_wq_df_fil) # all samples
outputs_df_unf = create_outputs(abs_wq_df_unf) # all samples
 
#%% make plots for all samples

make_plots(outputs_df,'Filtered and Unfiltered Samples')
make_plots(outputs_df_fil,'Filtered Samples')
make_plots(outputs_df_unf,'Unfiltered Samples')

#%% save output

outputs_df.to_csv(output_dir+'streams_RF_It0_results_3.csv',index=False)
   
#%% make and save output.

make_and_save_outputs(abs_wq_df,output_dir+'streams_PLS_It0-9_results.csv',
                      iterations = 10)

#%%

for i in np.linspace(5,10,5):
    print(i)
