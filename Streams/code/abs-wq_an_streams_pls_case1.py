# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:39:55 2021

@author: jbarrett.carter
"""
#Case 1: To test temporal variability, could past data return accurate results for the same sites?	
#Model will be trained on the hogdn and hogup values of the 2-year sampling
#Model will be tested on hogup and hogdn values of the new sampling
#First run will only look at orthophosphate (Phosphate-P)
			
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

#%% A: Set paths and bring in data; using Barrett's data only (hogup and hogdn only)

path_to_wqs = '/Users/ethanlantzy/Documents/GitHub/water_quality-spectroscopy' #for Laptop; path to relevant files
inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')                   #file path for input data folder
output_dir=os.path.join(path_to_wqs,'Streams/outputs/')                        #file path for outputs data folder

# abs_wq_df_fn = 'abs_wq_df_streams_clean.csv'                                 #input data; using Barrett's data only (hogup/hogdn)
# abs_wq_df_fn = 'abs_wq_df_streams_combined_clean.csv'                        #input data; using Barrett and Ethan's data (hogup/hogdn)
abs_wq_df_fn = 'abs_wq_df_streams_2023_clean.csv'                              #input data; using Ethan's data (all sites; Ocean Optics)
# abs_wq_df_fn = 'abs_wq_df_streams_2023_clean_SN.csv'                         #input data; using Ethan's data (all sites; StellarNet)
# abs_wq_df_fn = 'abs_wq_df_streams_all_clean.csv'                             #input data; using Barrett and Ethan's data (hogdn/hogup for Barrett, all sites for Ethan)


abs_wq_df_fil=pd.read_csv(inter_dir+abs_wq_df_fn)                              #translate computer file into program variable

#%% B: Set paths and bring in data; use Barrett's data (hogup/hogdn) to train and my data to test

#path_to_wqs = '/Users/ethanlantzy/Documents/GitHub/water_quality-spectroscopy' #for Laptop; path to relevant files
#inter_dir=os.path.join(path_to_wqs,'Streams/intermediates/')                   #file path for input data folder
#output_dir=os.path.join(path_to_wqs,'Streams/outputs/')                        #file path for outputs data folder
#input_train_fn = 'abs_wq_df_streams.csv'                                    #input data file for training
#input_train_df = pd.read_csv(inter_dir+input_train_fn)                   #translate computer file into dataframe

#input_test_fn_excel = 'abs_wq_df_streams_OO_Ethan.xlsx'                     #name of input Excel file for testing
#input_test_df = pd.read_excel(inter_dir+input_test_fn_excel)             #read the Excel file into a dataframe

#%% A and C: subset by filtration and sampling site; use Barrett's data to train and test OR use combined dataframe

#abs_wq_df_fil = abs_wq_df.loc[abs_wq_df['Filtered']==True,:]                  #clean df is already filtered
#abs_wq_df_fil = abs_wq_df_fil.loc[abs_wq_df_fil.Name.isin(['hogdn', 'hogup'])]#clean df is already site-specific
input_df = abs_wq_df_fil                                                       #define variable as the input for future functions

species = abs_wq_df_fil.columns[0:3]                                               #create list from column headings
s = species[2]                                                                 #defines the third element from the list of column headings as 's'
species = ['Nitrate-N']                                                        #creates list for variable 'species'

#%% B: subset by filtration and sampling site; use Barrett's dataframe to train and my dataframe to test
       
#input_train_df_fil = input_train_df.loc[input_train_df['Filtered']==True,:]                       #used to include filtered samples
#input_train_df_fil = input_train_df_fil.loc[input_train_df_fil.Name.isin(['hogdn', 'hogup'])]     #for site-based subsetting
#input_train_df = input_train_df_fil                                                               #define variable as the training input for future functions

#species = input_train_df.columns[0:8]                                          #create list from first eight column names
#s = species[2]                                                                 #defines the third element from the list of column headings as 's'
#species = ['Nitrate-N']                                                        #redefine species as [see inside brackets]

#input_test_df_fil = input_test_df.loc[input_test_df.Name.isin(['hogdn', 'hogup'])] #for site-based subsetting; this dataset is already filtered
#input_test_df = input_test_df_fil                                                  #define variable as the testing input for future functions

#species = input_test_df.columns[0:3]                                           #create list from first three column names
#s = species[2]                                                                 #defines the third element from the list of column headings as 's'
#species = ['Nitrate-N']                                                        #redefine species as [see inside brackets

#%% Create function for writing outputs                                        #Define function for creating outputs (to use later)

def create_outputs(input_df,iterations = 1, autosave = False, return_df = False, 
                   return_all = False, output_path = None, subset_name = '2023_data'):
    #input traindf and test df
    def write_output_df(the_output,output_name,species_name,iteration_num):
    
        if isinstance(the_output,float):                                       #place output in data frame depending on if it is of float or list form
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
    
    outputs_df = pd.DataFrame(columns= ['output','species','iteration','value']) #save outputs in dataframe
    
    output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
                    'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
                    'train_rmse','test_mape','train_mape','n_comp']
    
    variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                      'list(X_test.index)','list(X_train.index)','r_sq','r_sq_train','RMSE_test',
                      'RMSE_train','MAPE_test','MAPE_train','n_comp']
    
       
    iteration = 1                                                              #this is for testing
    
    if type(iterations)==int:                                                  #check if the variable is of an integer type
        
        iterations = range(iterations)                                         #if so, the variable is converted to a range object
    
    for s in species:                                                          #starts for loop in which s takes on each value in the list species
        
        for iteration in iterations:
            print('Analyzing '+s)                                              #will print Analyzing (chemical name) each iteration
            print('Iteration - '+str(iteration))                               #will print Interation - (iteration #) each iteration
            
            Y = input_df[s]                                                    #extract a specific column ('s') from the data frame
            keep = Y>0                                                         #filter extraction to only keep values that are greater than zero 
            
            inter_df = input_df.loc[keep,:]                                    #filter rows in 'input_df' to columns found in the previous lines
            
            X = inter_df.loc[:,'band_1':'band_1024']                           #extract a subset of inter_df for all rows and columns from 'band_1' to 'band_1024'.
            
            Y = inter_df[s]                                                    #extract the column specified by s
            
            X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=iteration, #split into 30% test and 70% training set
                                                                    test_size = 0.3)
            
            param_grid = [{'n_components':np.arange(1,20)}]                    #defines grid for hyperparameter to be tuned
            pls = PLSRegression() #create PLSRegression
            clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_absolute_error') #perform hyperparameter tuning for pls model; search parameter grid and identify best using neg. MAE
            
            clf.fit(X_train,y_train)                                           #fit model to data to find best parameters
            n_comp = float(clf.best_params_['n_components'])                   #extract optimal # of components from the best parameters
            pls_opt = clf.best_estimator_                                      #extract best model (which includes best hyperparameters)
            Y_hat = list(pls_opt.predict(X_test)[:,0])                         #testing set predictions using the best model
            Y_hat_train = list(pls_opt.predict(X_train)[:,0])                  #training set predictions using the best model
            
            r_sq = float(pls_opt.score(X_test,y_test))                         #r^2 for testing set
            r_sq_train = float(pls_opt.score(X_train,y_train))                 #r^2 for training set
    
            MSE_test = MSE(y_test,Y_hat)                                       #calculate MSE by comparing actual (y_test) to predicted (Y_hat_) --> test set
            RMSE_test = float(np.sqrt(MSE_test))                               #takes square root of MSE; converts type to float --> test set
            
            MSE_train = MSE(y_train,Y_hat_train)                               #calculate MSE by comparing actual (y_test) to predicted (Y_hat_) --> training set
            RMSE_train = float(np.sqrt(MSE_train))                             #takes square root of MSE; converts type to float --> training set
            
            abs_test_errors = abs(y_test-Y_hat)                                #absolute errors for testing set
            APE_test = abs_test_errors/y_test                                  #APE = absolute percent error,decimal (absolute error as a percentage of actual values)
            MAPE_test = float(np.mean(APE_test)*100)                           #this is percentage
            
            abs_train_errors = abs(y_train-Y_hat_train)                        #difference between training and test values (absolute values)
            APE_train = abs_train_errors/y_train                               #APE = Absolute Percent Error, decimal
            MAPE_train = float(np.mean(APE_train)*100)                         #MAPE: Mean Average Percent Error, percentage
            
            for out in range(len(output_names)):                               #for each variable,
                # print(out)
                sub_df = write_output_df(eval(variable_names[out]), output_names[out], s, iteration) #assign a variable name, value, species, and iteration
                # outputs_df = outputs_df.append(sub_df,ignore_index=True)
                
                outputs_df = pd.concat([outputs_df,sub_df],ignore_index=True)  #concatenate the separate assignments into one table
                
            filename = 'pls_streams-2023_data_filtered_PLS_It0-19.joblib'      #provide filename for the document
            pickle_path = os.path.join(output_dir,'picklejar',filename)        #send it to the pickejar folder with the given filename
            dump(clf,pickle_path)                                              #save the model to files
            
            if autosave == True:
                
                outputs_df.to_csv(output_path,index=False)                     #saves it as a csv file
      
    if return_df:
        
      return(outputs_df)                                                       #return full dataframe
      
    if return_all:                                                             #return specific sections of the dataframe
        
        return({'outputs_df':outputs_df, 'X_train':X_train, 'y_train':y_train,
                'inter_df':inter_df})

#%% Define function for making plots                                           #Define function for making plots (to use later)

def make_plots(outputs_df, output_label):
    
    fig, ax = plt.subplots(dpi=300)  # Create a single subplot
    fig.set_size_inches(10, 12)  # Set the size of the figure
    fig.suptitle(output_label, fontsize=18)  # Add a centered title with a fontsize of 18

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

        line11 = np.linspace(min(np.concatenate((y_true_train, y_hat_train,
                                                y_true_test, y_hat_test))),
                             max(np.concatenate((y_true_train, y_hat_train,
                                                y_true_test, y_hat_test))))

        y_text = min(line11) + (max(line11) - min(line11)) * 0
        x_text = max(line11) - (max(line11) - min(line11)) * 0.5

        train_rsq = outputs_df['value'][(outputs_df.output == 'train_rsq') &
                                        (outputs_df.species == s)]

        train_rsq = np.mean(train_rsq)

        test_rsq = outputs_df['value'][(outputs_df.output == 'test_rsq') &
                                       (outputs_df.species == s)]

        test_rsq = np.mean(test_rsq)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)

        ax.plot(y_true_train, y_hat_train, 'o', markersize=4, label='training set')
        ax.plot(y_true_test, y_hat_test, 'o', markersize=4, label='test set')
        ax.plot(line11, line11, 'k--', label='1:1 line')
        ax.legend(loc='upper left', fontsize=16)
        ax.set_xlabel('Lab Measured ' + s + ' (mg/L)', fontsize=16)
        ax.set_ylabel('Predicted ' + s + ' (mg/L)', fontsize=16)
        ax.text(x_text, y_text, r'$train\/r^2 =$' + str(np.round(train_rsq, 3)) + '\n'
                + r'$test\/r^2 =$' + str(np.round(test_rsq, 3)), fontsize=16)

        # Add vertical lines with legend labels
        ax.axvline(x=0.053, linestyle='--', color='black', label='MDL')
        ax.axvline(x=0.131, linestyle='--', color='gray', label='PQL')

        ax.legend(loc='upper left', fontsize=16)
        ax.set_xlabel('Lab Measured ' + s + ' (mg/L)', fontsize=16)
        ax.set_ylabel('Predicted ' + s + ' (mg/L)', fontsize=16)
        ax.text(x_text, y_text, r'$train\/r^2 =$' + str(np.round(train_rsq, 3)) + '\n'
                + r'$test\/r^2 =$' + str(np.round(test_rsq, 3)), fontsize=16)
    plt.show()

#%% create outputs for models trained with samples and save to exported file (use the create_outputs function)

outputs_df = create_outputs(abs_wq_df_fil, iterations = 80, autosave = True, #filtered samples, no synthetic samples
               output_path = os.path.join(output_dir,'streams_PLS_results.csv'),
               return_df = True) #create outputs and save

 
#%% make plots for all samples and show (use the make_plots function)

make_plots(outputs_df,'Comparison of Measured and Predicted Values of Nitrate-N')

#%% filter out predcited values less than the MDL and re-make the plots and R^2

# Specify the path where you want to save the Excel file
excel_path = os.path.join(output_dir, '80_It_Results.xlsx')

# Export the DataFrame to Excel
outputs_df.to_excel(excel_path, index=False)
