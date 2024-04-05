# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:39:55 2021

@author: jbarrett.carter
"""

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

#for looking up available scorers
# import sklearn.metrics
# sorted(sklearn.metrics.SCORERS.keys())

from joblib import dump

#%% Set paths and bring in data

user = os.getlogin() 
# path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
# path_to_wqs = 'C:\\Users\\'+ user + '\\Documents\\GitHub\\PhD\\water_quality-spectroscopy' #for work computer
path_to_wqs = 'D:\\GitHub\\PhD\\water_quality-spectroscopy' #external HD
# path_to_wqs = '/blue/ezbean/jbarrett.carter/water_quality-spectroscopy/' # for HiPerGator
inter_dir=os.path.join(path_to_wqs,'Hydroponics/intermediates/')
output_dir=os.path.join(path_to_wqs,'Hydroponics/outputs/')

abs_wq_df_fn = 'abs-wq_HNSrd30_df.csv' # for diluted samples
# abs_wq_df_fn = 'abs-wq_HNSr_df.csv' # for undiluted samples

syn_abs_wq_df_fn = 'abs-wq_HNSsd30_df.csv' # for diluted synthetic samples
# syn_abs_wq_df_fn = 'abs-wq_HNSs_df.csv' # for undiluted synthetic samples

# abs_wq_df_dil_fn = 'abs-wq_HNSsd30_df.csv' # for diluted synthetic samples by themselves
# abs_wq_df_und_fn = 'abs-wq_HNSs_df.csv' # for undiluted synthetic samples by themselves

# Bring in data
abs_wq_df=pd.read_csv(inter_dir+abs_wq_df_fn)

# abs_wq_df_dil=pd.read_csv(inter_dir+abs_wq_df_dil_fn)
# abs_wq_df_und=pd.read_csv(inter_dir+abs_wq_df_und_fn)
# abs_wq_df = abs_wq_df.loc[0:56,:] # because the minimum of the two sample sizes (diluted versus undiluted) is 56

syn_abs_wq_df=pd.read_csv(inter_dir+syn_abs_wq_df_fn)

subset_name = abs_wq_df_fn.split(sep = '_')[1]
                             
#%% Create function for writing outputs

def create_outputs(input_df,iterations = 1, autosave = False, output_path = None,
                   subset_name = None, syn_aug = False, syn_df = None,
                   species = 'All'):
    
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
    s = 'Molybdenum' # this is for testing
    
    outputs_df = pd.DataFrame(columns= ['output','species','iteration','value']) #save outputs in dataframe
    
    output_names = ['y_hat_test','y_hat_train','y_true_train','y_true_test',
                    'test_ind','train_ind','test_rsq','train_rsq','test_rmse',
                    'train_rmse','test_mape','train_mape','n_comp']
    
    variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                      'list(X_test.index)','list(X_train.index)','r_sq','r_sq_train','RMSE_test',
                      'RMSE_train','MAPE_test','MAPE_train','n_comp']
    
    
    
    iteration = 1 # this is for testing
    
    if species == 'All':
    
        species = input_df.columns[0:14]
    
    for s in species:
        for iteration in range(iterations):
            print('Analyzing '+s)
            print('Iteration - '+str(iteration))
            
            samp_size = 50 # to be consistent with streams
            
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
                        
            if syn_aug:
                
                syn_samp_size = 46
                
                if syn_df.shape[0]>syn_samp_size:
                
                    syn_df = syn_df.sample(n = syn_samp_size, random_state = iteration)
                    
                X_syn = syn_df.loc[:,'band_1':'band_1024']
                
                Y_syn = syn_df[s]
                
                X_train = pd.concat([X_train,X_syn],ignore_index = True)
                y_train = pd.concat([y_train,Y_syn],ignore_index = True)
            
            param_grid = [{'n_components':np.arange(1,20)}]
            pls = PLSRegression()
            clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_absolute_error')

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
            
            for out in range(len(output_names)):
                # print(out)
                sub_df = write_output_df(eval(variable_names[out]), output_names[out], s, iteration)
                outputs_df = pd.concat([outputs_df,sub_df],ignore_index=True)
            
            # filename = f'pls_{subset_name}_syn-aug-{syn_aug}_{s}_It{iteration}.joblib'
            # pickle_path = os.path.join(output_dir,'picklejar',filename)
            # dump(clf,pickle_path)
            
            if autosave == True:
                
                outputs_df.to_csv(output_path,index=False)
                
    return(outputs_df)

#%% Define function for making plots

def make_plots(outputs_df, output_label):
        
    fig, axs = plt.subplots(5,3)
    fig.set_size_inches(20,30)
    fig.suptitle(output_label,fontsize = 16)
    fig.tight_layout(pad = 2)
    axs[4, 2].axis('off')
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
        if (row == 0 and col == 0):
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

#%% make and save outputs

# def make_and_save_outputs(input_df,output_path,iterations = 1):
#     outputs_df = create_outputs(input_df,iterations)
#     outputs_df.to_csv(output_path,index=False)
# #     make_plots(outputs_df,'Hydroponics')
# #     return(outputs_df)

#%% Run create_outputs function for testing

outputs_r_df = create_outputs(abs_wq_df, iterations = 20, autosave = True,
                            output_path = os.path.join(output_dir,'HNSd30-SO4_syn-aug-False_PLS_It0-19_results.csv'),
                            subset_name = subset_name,syn_aug = False,
                            species = ['Sulfate']) # no synthetic samples



outputs_rs_df = create_outputs(abs_wq_df, iterations = 20, autosave = True,
                            output_path = os.path.join(output_dir,'HNSd30-SO4_syn-aug-True_PLS_It0-19_results.csv'),
                            subset_name = subset_name,syn_aug = True, syn_df = syn_abs_wq_df,
                            species = ['Sulfate']) # real and synthetic samples

make_plots(outputs_r_df,'Hydroponic Samples - Real only')
make_plots(outputs_rs_df,'Hydroponic Samples - Real and Synthetic')

# create_outputs(abs_wq_df_dil, iterations = 20, autosave = True,
#                output_path = os.path.join(output_dir,'HNSsd30_PLS_It0-19_results.csv'),
#                syn_aug = False) # diluted synthetic samples by themselves

# create_outputs(abs_wq_df_und, iterations = 20, autosave = True,
#                output_path = os.path.join(output_dir,'HNSs_PLS_It0-19_results.csv'),
#                syn_aug = False) # diluted synthetic samples by themselves
 
#%% make plots for all samples

# make_plots(outputs_df,'Hydroponic Samples')

#%% save output

# outputs_df.to_csv(output_dir+'streams_PLS_B10_results.csv',index=False)
   
#%% make and save output.

# make_and_save_outputs(abs_wq_df,output_dir+'HNSr_PLS_It0-19_results.csv',
#                       iterations = 20)