# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:39:55 2021

@author: jbarrett.carter
"""
# Case 1: Two-year data set (hogdn samples only) used to train model & model tested on 50 new samples
# Edited by Ethan Lantzy in 2023-24
#%% import libraries

import pandas as pd
import numpy as np
import os
# import datetime as dt
import matplotlib.pyplot as plt
# import scipy
# from scipy import stats
# import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.utils import resample
from sklearn.metrics import mean_squared_error as MSE
# from sklearn.ensemble import RandomForestRegressor

# for looking up available scorers
# import sklearn.metrics
# sorted(sklearn.metrics.SCORERS.keys())

from joblib import dump

#%% Set paths and bring in data

path_to_wqs = '/Users/ethanlantzy/Documents/GitHub/water_quality-spectroscopy' # for Laptop; path to relevant files
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/') #file path for input data folder
output_dir=os.path.join(path_to_wqs,'Streams/outputs/') #file path to send results folder

abs_wq_df_fn = 'abs_wq_df_streams.csv' #input data file (absorbance and lab results)

abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn) #translate computer file into program variable

#%% seperate into filtered and unfiltered sample sets; subset by sampling site

abs_wq_df_fil = abs_wq_df.loc[abs_wq_df['Filtered']==True,:] #Used to filter for true samples

# abs_wq_df_fil = abs_wq_df_fil.loc[abs_wq_df_fil.Name.isin(['hogdn'])] # for site-based subsetting

input_df = abs_wq_df_fil # define variable as the input for future functions

species = abs_wq_df.columns[0:8] # create list from column headings
s = species[2]                   # defines the third element from the list of column headings as 's'
species = ['Nitrate-N', 'OP']    # ensure names and column headers match
                             
#%% Create function for writing outputs

def create_outputs(input_df,iterations = 1, autosave = False, return_df = False, 
                   return_all = False, output_path = None, subset_name = 'hogdn_only'):
    
    def write_output_df(the_output,output_name,species_name,iteration_num):
    
        if isinstance(the_output,float): #Place output in data frame depending on if it is of float or list form
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
    # s = 'Molybdenum' # this is for testing
    
    outputs_df = pd.DataFrame(columns= ['output','species','iteration','value']) #save outputs in dataframe
    
    output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
                    'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
                    'train_rmse','test_mape','train_mape','n_comp']
    
    variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                      'list(X_test.index)','list(X_train.index)','r_sq','r_sq_train','RMSE_test',
                      'RMSE_train','MAPE_test','MAPE_train','n_comp']
    
       
    iteration = 1 # this is for testing
    
    if type(iterations)==int: #check if the variable is of an integer type
        
        iterations = range(iterations) #if so, the variable is converted to a range object
    
    for s in species: #starts for loop in which s takes on each value in the list species
        
        for iteration in iterations:
            print('Analyzing '+s) #will print Analyzing (chemical name) each iteration
            print('Iteration - '+str(iteration)) #will print Interation - (iteration #) each iteration
            
            Y = input_df[s] #extract a specific column ('s') from the data frame
            keep = Y>0 #filter extraction to only keep values that are greater than zero 
            
            inter_df = input_df.loc[keep,:] #Filter rows in 'input_df' to columns found in the previous lines
            
            if sum(keep)>samp_size: #randomly select certain values if sample size is limiting
            
                inter_df = inter_df.sample(n = samp_size, random_state = iteration) #randomly sample rows from inter_df
            
            X = inter_df.loc[:,'band_1':'band_1024'] #Extract a subset of inter_df for all rows and columns from 'band_1' to 'band_1024'.
            
            Y = inter_df[s] #Extract the column specified by s
            
            X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=iteration, #Split into 30% test and 70% training set
                                                                test_size = 0.3)
            
            param_grid = [{'n_components':np.arange(1,20)}] #defines grid for hyperparameter to be tuned
            pls = PLSRegression() #create PLSRegression
            clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_absolute_error') #perform hyperparameter tuning for pls model; search parameter grid and identify best using neg. MAE
            
            clf.fit(X_train,y_train) #fit model to data to find best parameters
            n_comp = float(clf.best_params_['n_components']) #Extract optimal # of components from the best parameters
            pls_opt = clf.best_estimator_ #extract best model (which includes best hyperparameters)
            Y_hat = list(pls_opt.predict(X_test)[:,0]) #testing set predictions using the best model
            Y_hat_train = list(pls_opt.predict(X_train)[:,0]) #training set predictions using the best model
            
            r_sq = float(pls_opt.score(X_test,y_test)) #r^2 for testing set
            r_sq_train = float(pls_opt.score(X_train,y_train)) #r^2 for training set
    
            MSE_test = MSE(y_test,Y_hat) #calculate MSE by comparing actual (y_test) to predicted (Y_hat_) --> test set
            RMSE_test = float(np.sqrt(MSE_test)) #takes square root of MSE; converts type to float --> test set
            
            MSE_train = MSE(y_train,Y_hat_train) #calculate MSE by comparing actual (y_test) to predicted (Y_hat_) --> training set
            RMSE_train = float(np.sqrt(MSE_train)) #takes square root of MSE; converts type to float --> training set
            
            abs_test_errors = abs(y_test-Y_hat) #absolute errors for testing set
            APE_test = abs_test_errors/y_test # APE = absolute percent error,decimal (absolute error as a percentage of actual values)
            MAPE_test = float(np.mean(APE_test)*100) # this is percentage
            
            abs_train_errors = abs(y_train-Y_hat_train) #difference between training and test values (absolute values)
            APE_train = abs_train_errors/y_train # APE = Absolute Percent Error, decimal
            MAPE_train = float(np.mean(APE_train)*100) #MAPE: Mean Average Percent Error, percentage
            
            for out in range(len(output_names)):
                # print(out)
                sub_df = write_output_df(eval(variable_names[out]), output_names[out], s, iteration)
                # outputs_df = outputs_df.append(sub_df,ignore_index=True)
                
                outputs_df = pd.concat([outputs_df,sub_df],ignore_index=True)
                
            filename = 'pls_streams-hogdn_only-syn-aug-FALSE_PLS_It0-19.joblib'
            pickle_path = os.path.join(output_dir,'picklejar',filename)
            dump(clf,pickle_path)
            
            if autosave == True:
                
                outputs_df.to_csv(output_path,index=False)
      
    if return_df:
        
      return(outputs_df)
      
    if return_all:
        
        return({'outputs_df':outputs_df, 'X_train':X_train, 'y_train':y_train,
                'inter_df':inter_df})

#%% Define function for making plots

def make_plots(outputs_df, output_label):

    ## make plots for both filtered and unfiltered samples
        
    fig, axs = plt.subplots(1,2,dpi = 300)
    fig.set_size_inches(15,15)
    fig.suptitle(output_label,fontsize = 18)
    fig.tight_layout(pad = 4)
    #axs[2, 2].axis('off') #This will create a 3x3 matrix (Python starts index at 0)
    #row = 0 #For one row/two columns, only one variable is used
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
        
        ax = axs[col]
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)
        
        axs[col].plot(y_true_train,y_hat_train,'o',markersize = 4, label = 'training set')
        axs[col].plot(y_true_test,y_hat_test,'o',markersize = 4, label = 'test set')
        axs[col].plot(line11,line11,'k--',label= '1:1 line')
        # axs[row,col].set_title(s)
        axs[col].legend(loc = 'upper left',fontsize = 16)
        axs[col].set_xlabel('Lab Measured '+s+' (mg/L)',fontsize = 16)
        axs[col].set_ylabel('Predicted '+s+' (mg/L)',fontsize = 16)
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
        
        #if col == 1:
            #col = 0
            #row += 1
        #else
        col +=1
        fig.show()

#%% function for make and save outputs #Not necessary --> have autosave in next section set as True

# def make_and_save_outputs(input_df,output_path,iterations = 1):
#     outputs_df = create_outputs(input_df,iterations)
#     outputs_df.to_csv(output_path,index=False)

#%% Create outputs for models trained with filtered, unfiltered, and all samples


### Sites experiment ###

 #for name in names:

outputs_dict = create_outputs(abs_wq_df_fil, iterations = 20, autosave = True,
               output_path = os.path.join(output_dir,'pls_streams-hogdn_only-syn-aug-FALSE_PLS_It0-19.joblib')
               ,syn_aug = False) # filtered samples, no synthetic samples

### Filtration Experiment ###

# create_outputs(abs_wq_df_fil, iterations = 20, autosave = True,
#                 output_path = os.path.join(output_dir,'streams-fil_PLS_It0-19_results.csv'),
#                 subset_name = 'fil') # filtered samples

# create_outputs(abs_wq_df_unf, iterations = 20, autosave = True,
#                 output_path = os.path.join(output_dir,'streams-unf_PLS_It0-19_results.csv'),
#                 subset_name = 'unf') # unfiltered samples

# create_outputs(abs_wq_df, iterations = 20, autosave = True,
#                 output_path = os.path.join(output_dir,'streams-comb_PLS_It0-19_results.csv'),
#                 subset_name = 'comb') # combined samples for filtration experiment

# create_outputs(abs_wq_df, iterations = 20, autosave = True,
#                 output_path = os.path.join(output_dir,'streams-comb_PLS_It0-19_results.csv'),
#                 subset_name = 'comb') # combined samples for filtration experiment

### Synthetic Samples Experiment ###


### For Testing ###

#outputs_dict = create_outputs(abs_wq_df_fil, iterations = 1, autosave = False,
               # return_all = True, syn_aug = True, syn_df = syn_abs_wq_df)

 
#%% make plots for all samples

outputs_df = outputs_dict['outputs_df']

# make_plots(outputs_df,'Filtered and Unfiltered Samples')
make_plots(outputs_df,'Filtered and Synthetic Samples')
# make_plots(outputs_df_unf,'Unfiltered Samples')

#%% save output

# outputs_df.to_csv(output_dir+'streams_PLS_B10_results.csv',index=False)
   
#%% make and save output.

# make_and_save_outputs(abs_wq_df,output_dir+'streams_PLS_It10-19_results.csv',
#                       iterations = np.arange(10,20))
