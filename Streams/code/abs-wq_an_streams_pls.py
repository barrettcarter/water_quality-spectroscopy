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
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor

#for looking up available scorers
import sklearn.metrics
sorted(sklearn.metrics.SCORERS.keys())

#%%

### Set paths and bring in data

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
                             
#%% To Test
#################################################################

# ### Tuning the models

# ## Nitrate
# ## PlSR

# for rs in range(10):

#     param_grid = [{'n_components':np.arange(1,20)}]
    
#     # keep = abs_wq_df['Name'].isin(['hogdn','hat'])
#     # X = abs_wq_df.loc[:,'band_1':'band_1024'].to_numpy()
#     Y = abs_wq_df['Nitrate-N']
#     keep = pd.notna(Y)
#     X = abs_wq_df.loc[keep,'band_1':'band_1024']
#     Y = Y[keep]
    
#     # name_dum = pd.get_dummies(abs_wq_df['Name'])
#     # filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
#     # X = abs_wq_df.loc[keep,:]
#     # Y = abs_wq_df.Nitrate[keep].to_numpy()
#     # name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
#     # X = pd.concat([name_dum,X],axis=1).to_numpy()
#     # X = name_dum
    
    
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=rs,
#                                                         test_size = 0.3)
#     # test_names = X_test.Name.reset_index(drop=True)
#     # test_filt = X_test.Filtered.reset_index(drop=True)
    
#     # X_train = X_train.loc[:,'band_1':'band_1024']
#     # X_test = X_test.loc[:,'band_1':'band_1024']
#     pls = PLSRegression()
#     clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_absolute_error')
#     # clf = GridSearchCV(pls,param_grid,scoring = 'r2')
#     # clf.cv_results_
#     # clf = GridSearchCV(pls,param_grid)
#     clf.fit(X_train,y_train)
#     n_comp = clf.best_params_['n_components']
#     print('Number of components:\t'+str(n_comp))
#     pls_opt = clf.best_estimator_
#     Y_hat = pls_opt.predict(X_test)
#     Y_hat_train = pls_opt.predict(X_train)
    
#     r_sq = pls_opt.score(X_test,y_test)
#     print('Test r-squared value:\t'+str(round(r_sq,3)))
    
#     r_sq_train = pls_opt.score(X_train,y_train)
#     print('Training r-squared value:\t'+str(round(r_sq_train,3)))
    
    
    
#     ### Plot results
#     # plt.plot(Y_hat,y_test,'b.')
    
#     line11 = np.linspace(min(np.concatenate((y_test,Y_hat[:,0]))),
#                          max(np.concatenate((y_test,Y_hat[:,0]))))
    
#     # lr = LinearRegression().fit(Y_hat,y_test)
#     # linelr = lr.predict(line11.reshape(-1,1))
    
#     plt.figure()
#     plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
#     plt.plot(line11,line11,label= '1:1 line')
#     # plt.plot(line11,linelr,label = 'regression line')
#     plt.xlabel('Lab Measured Nitrate (mg/L)')
#     plt.ylabel('Predicted Nitrate (mg/L)')
#     plt.text(0.8*max(line11),min(line11),r'$r^2 =$'+str(np.round(r_sq,3)))
#     plt.title('Test Set')
#     plt.legend()
#     plt.show()
    
#     line11 = np.linspace(min(np.concatenate((y_train,Y_hat_train[:,0]))),
#                          max(np.concatenate((y_train,Y_hat_train[:,0]))))
    
#     # lr = LinearRegression().fit(Y_hat,y_test)
#     # linelr = lr.predict(line11.reshape(-1,1))
    
#     plt.plot(y_train,Y_hat_train,'o',markersize = 4, label = 'predictions')
#     plt.plot(line11,line11,label= '1:1 line')
#     # plt.plot(line11,linelr,label = 'regression line')
#     plt.title('Training Set')
#     plt.xlabel('Lab Measured Nitrate (mg/L)')
#     plt.ylabel('Predicted Nitrate (mg/L)')
#     plt.text(0.8*max(line11),min(line11),r'$r^2 =$'+str(np.round(r_sq_train,3)))
#     plt.legend()
#     plt.show()

#%%

# # make better plot
# data_out = pd.DataFrame({'y_test':y_test,'y_pred':Y_hat[:,0]})
# data_out = pd.concat([data_out,test_names,test_filt],axis=1)

# sns.set_theme(style ='ticks',font_scale = 1.25,
#               palette = 'colorblind')

# g = sns.relplot(
#     data=data_out,
#     x = 'y_test',
#     y = 'y_pred',
#     hue = 'Filtered',
#     style = 'Name',
#     s = 60
#     )

# plt.plot(line11,line11,label= '1:1 line',color = 'k',ls = 'dashed')
# plt.xlabel('Lab Measured Nitrate (mg/L)')
# plt.ylabel('Predicted Nitrate (mg/L)')
# plt.text(0.5,2,r'$r^2 =$'+str(np.round(r_sq,3)))


#%% Create function for writing outputs

def create_outputs(input_df):
    
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
                    'train_rmse','test_mape','train_mape','n_comp']
    
    variable_names = ['Y_hat','Y_hat_train','list(y_train)', 'list(y_test)',
                      'list(X_test.index)','list(X_train.index)','r_sq','r_sq_train','RMSE_test',
                      'RMSE_train','MAPE_test','MAPE_train','n_comp']
    
    # for vn in variable_names: # this is for testing
    #     print(vn+':',type(eval(vn)))
    
    
    ## save outputs in dictionaries
    # y_hat_test_dict = dict()
    # y_hat_train_dict = dict()
    # # y_true_train_dict = dict()
    # # y_true_test_dict = dict()
    # test_ind_dict = dict()
    # train_ind_dict = dict()
    # test_rsq_dict = dict()
    # train_rsq_dict = dict()
    # test_rmse_dict = dict()
    # train_rmse_dict = dict()
    # test_mape_dict = dict()
    # train_mape_dict = dict()
    # n_comp_dict = dict()
    
    # for s in species:
    #     y_hat_test_dict[s]=[]
    #     y_hat_train_dict[s] = []
    #     test_ind_dict[s] = []
    #     train_ind_dict[s] = []
    #     test_rsq_dict[s] = []
    #     train_rsq_dict[s] = []
    #     test_rmse_dict[s] = []
    #     train_rmse_dict[s] = []
    #     test_mape_dict[s] = []
    #     train_mape_dict[s] = []
    #     n_comp_dict[s] = []
    
    iteration = 1 # this is for testing
    
    species = input_df.columns[1:6]
    
    for s in species:
        Y = input_df[s]
        keep = pd.notna(Y)
        X = input_df.loc[keep,'band_1':'band_1024']
        Y = Y[keep]
        
        # name_dum = pd.get_dummies(abs_wq_df['Name'])
        # filtered_dum = pd.get_dummies(abs_wq_df['Filtered'])
        # X = abs_wq_df.loc[keep,:]
        # Y = abs_wq_df.Nitrate[keep].to_numpy()
        # name_dum = pd.get_dummies(abs_wq_df['Name'][keep])
        # X = pd.concat([name_dum,X],axis=1).to_numpy()
        # X = name_dum
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=iteration,
                                                            test_size = 0.3)
        # test_names = X_test.Name.reset_index(drop=True)
        # test_filt = X_test.Filtered.reset_index(drop=True)
        
        # X_train = X_train.loc[:,'band_1':'band_1024']
        # X_test = X_test.loc[:,'band_1':'band_1024']
        param_grid = [{'n_components':np.arange(1,20)}]
        pls = PLSRegression()
        clf = GridSearchCV(pls,param_grid,scoring = 'neg_mean_absolute_error')
        # clf.cv_results_
        # clf = GridSearchCV(pls,param_grid)
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
        
        
        # y_hat_test_dict[s].extend(list(Y_hat))
        # y_hat_train_dict[s].extend(list(Y_hat_train))
        # test_ind_dict[s].extend(list(X_test.index))
        # train_ind_dict[s].extend(list(X_train.index))
        # test_rsq_dict[s].append(float(r_sq))
        # train_rsq_dict[s].append(float(r_sq_train))
        # test_rmse_dict[s].append(float(RMSE_test))
        # train_rmse_dict[s].append(float(RMSE_train))
        # test_mape_dict[s].append(float(MAPE_test))
        # train_mape_dict[s].append(float(MAPE_train))
        # n_comp_dict[s].append(float(n_comp))
        
        # sub_df = pd.DataFrame(columns= ['output','species','iteration','value'])
        # sub_df['value']=r_sq
        # sub_df['output']='test_rsq'
        # sub_df['species']=s
        # sub_df['iteration']=iteration
            
        # sub_df = write_output_df(r_sq, 'test_rsq', s, iteration)
        # outputs_df = outputs_df.append(sub_df,ignore_index=True)
        
        for out in range(len(output_names)):
            # print(out)
            sub_df = write_output_df(eval(variable_names[out]), output_names[out], s, iteration)
            outputs_df = outputs_df.append(sub_df,ignore_index=True)
        
        # plt.plot(Y_hat,y_test,'b.')
        
        # line11 = np.linspace(min(np.concatenate((y_test,Y_hat))),
        #                      max(np.concatenate((y_test,Y_hat))))
        
        # y_text1 = min(line11)+(max(line11)-min(line11))*0.05
        # y_text2 = min(line11)+(max(line11)-min(line11))*0.15
        # x_text = max(line11)-(max(line11)-min(line11))*0.3
        
        # # lr = LinearRegression().fit(Y_hat,y_test)
        # # linelr = lr.predict(line11.reshape(-1,1))
        
        # plt.plot(y_test,Y_hat,'o',markersize = 4, label = 'predictions')
        # plt.plot(line11,line11,label= '1:1 line')
        # plt.title('Test Set')
        # # plt.plot(line11,linelr,label = 'regression line')
        # plt.xlabel('Lab Measured '+s+' (mg/L)')
        # plt.ylabel('Predicted '+s+' (mg/L)')
        # plt.text(x_text,y_text1,r'$r^2 =$'+str(np.round(r_sq,3)))
        # plt.text(x_text,y_text2,r'MAPE = '+str(np.round(MAPE_test,1))+'%')
        # plt.legend()
        # plt.show()
        
        # line11 = np.linspace(min(np.concatenate((y_train,Y_hat_train))),
        #                      max(np.concatenate((y_train,Y_hat_train))))
        
        # y_text1 = min(line11)+(max(line11)-min(line11))*0.05
        # y_text2 = min(line11)+(max(line11)-min(line11))*0.15
        
        # x_text = max(line11)-(max(line11)-min(line11))*0.3
        
        # # lr = LinearRegression().fit(Y_hat,y_test)
        # # linelr = lr.predict(line11.reshape(-1,1))
        
        # plt.plot(y_train,Y_hat_train,'o',markersize = 4, label = 'predictions')
        # plt.plot(line11,line11,label= '1:1 line')
        # plt.title('Training Set')
        # # plt.plot(line11,linelr,label = 'regression line')
        # plt.xlabel('Lab Measured '+s+' (mg/L)')
        # plt.ylabel('Predicted '+s+' (mg/L)')
        # plt.text(x_text,y_text1,r'$r^2 =$'+str(np.round(r_sq_train,3)))
        # plt.text(x_text,y_text2,r'MAPE = '+str(np.round(MAPE_train,1))+'%')
        # plt.legend()
        # plt.show()
        
    return(outputs_df)

#%% Create outputs for models trained with filtered, unfiltered, and all samples

outputs_df = create_outputs(abs_wq_df) # all samples
outputs_df_fil = create_outputs(abs_wq_df_fil) # all samples
outputs_df_unf = create_outputs(abs_wq_df_unf) # all samples

    
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
make_plots(outputs_df_fil,'Filtered Samples')
make_plots(outputs_df_unf,'Unfiltered Samples')

#%% save plots

outputs_df.to_csv(output_dir+'streams_PLS_B1_results.csv',index=False)
