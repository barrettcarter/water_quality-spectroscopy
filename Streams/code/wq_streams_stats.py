# -*- coding: utf-8 -*-
"""
Created on Wed September 22 17:45 2021

@author: J. Barrett Carter
"""
#%% bring in libraries
import pandas as pd
import numpy as np
import os
# import datetime as dt
import matplotlib.pyplot as plt
# import scipy
from scipy import stats
import seaborn as sns
# from sklearn.linear_model import LinearRegression
#%% bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
wq_df_dir=os.path.join(path_to_wqs,'Streams/inputs/water_quality/')

wq_df_fn = 'wq_streams_aj_df.csv'
wq_df_fn2 = 'wq_streams_arl_df.csv'
wq_codes_fn = 'ARL_codes.csv'

# Bring in data
wq_df=pd.read_csv(wq_df_dir+wq_df_fn)
wq_df2=pd.read_csv(wq_df_dir+wq_df_fn2)
wq_codes = pd.read_csv(wq_df_dir+wq_codes_fn)


#%% some data wrangling

# clean up wq_df2
wq_df2 = wq_df2.loc[pd.notna(wq_df2['ARL_code']),:]
wq_df2.reset_index(drop = True,inplace = True)

# c = wq_df2.columns[1]
for c in wq_df2.columns:
    wq_df2.loc[:,c] = pd.to_numeric(wq_df2[c])

# Make units consistent

wq_df2.loc[:,'Phosphate-P'] = wq_df2.loc[:,'Phosphate-P']/1000
wq_df2.loc[:,'TP'] = wq_df2.loc[:,'TP']/1000

# select only relevant codes
wq_codes = wq_codes.iloc[0:wq_df2.shape[0],:]

# create sample IDs
wq_df['ID']=wq_df['Name']+wq_df['Date_col']

wq_df2['ID']=wq_codes['Name']+wq_codes['Date_col']
wq_df2['Name']=wq_codes['Name']
wq_df2['Date_col']=wq_codes['Date_col']
wq_df2['Filtered']=wq_codes['Filtered']

# mark problematic data from 11/5/2020
# samples from 11/5 and 11/19 were given same label and stored for too long

wq_df2['ID'][wq_codes['Date_col']=='11/5/2020']=\
wq_df2['ID'][wq_codes['Date_col']=='11/5/2020']+\
    wq_codes['Date_an'][wq_codes['Date_col']=='11/5/2020']


# re-order wq_df to match wq_df2

# separate filtered and unfiltered samples

wq_df2_fil = wq_df2.loc[wq_codes.Filtered==True,:]
wq_df2_unf = wq_df2.loc[wq_codes.Filtered==False,:]

# add ARL codes to wq_df separated by filtration
# row = 5
for row in range(wq_df.shape[0]):

    fil_code = wq_df2_fil.loc[wq_df2_fil['ID']==wq_df['ID'][row],'ARL_code'].values
    if fil_code.size != 1:
        wq_df.loc[row,'Fil_code'] = np.nan
        
    else:
        wq_df.loc[row,'Fil_code']=fil_code
        
    unf_code = wq_df2_unf.loc[wq_df2_unf['ID']==wq_df['ID'][row],'ARL_code'].values
    if unf_code.size != 1:
        wq_df.loc[row,'Unf_code'] = np.nan
        
    else:
        wq_df.loc[row,'Unf_code']=unf_code
        
wq_df = wq_df.loc[np.isnan(wq_df['Fil_code'])==False,:]

# pivot wq_df to make separate species columns

conc_df = pd.pivot_table(wq_df,values = 'Conc',columns = ['Species'],index = ['ID'])
conc_df.reset_index(inplace=True)

# separate into filtered and unfiltered samples for each lab
# the values are the same for lab1 because all samples were filtered
Lab1_fil = pd.pivot_table(wq_df,values = 'Conc',columns = ['Species'],index = ['Fil_code'])
Lab1_fil.reset_index(inplace=True)
Lab1_unf = pd.pivot_table(wq_df,values = 'Conc',columns = ['Species'],index = ['Unf_code'])
Lab1_unf.reset_index(inplace=True)

Lab2_fil = wq_df2.loc[wq_df2['ARL_code'].isin(Lab1_fil['Fil_code']),:]
Lab2_fil.reset_index(inplace=True)
Lab2_unf = wq_df2.loc[wq_df2['ARL_code'].isin(Lab1_unf['Unf_code']),:]
Lab2_unf.reset_index(inplace=True)

# remove unneeded columns

Lab1_fil = Lab1_fil.iloc[:,1:4]
Lab1_unf = Lab1_unf.iloc[:,1:4]
Lab2_fil = Lab2_fil.iloc[:,1:6]
Lab2_unf = Lab2_unf.iloc[:,1:6]


#%% make subsets

# Nit_fil = pd.DataFrame(columns = ['Lab1','Lab2'])
# Nit_fil['Lab1']=conc_df_fil['Nitrate-N']
# Nit_fil['Lab2']=wq_df2.loc[wq_df2['ARL_code'].isin(conc_df_fil['Fil_code']),'Nitrate-N'].values
# Nit_fil['Lab2']=pd.to_numeric(Nit_fil['Lab2'])

# Nit_unf = pd.DataFrame(columns = ['Lab1','Lab2'])
# Nit_unf['Lab1']=conc_df_unf['Nitrate-N']
# Nit_unf['Lab2']=wq_df2.loc[wq_df2['ARL_code'].isin(conc_df_unf['Unf_code']),'Nitrate-N'].values
# Nit_unf['Lab2']=pd.to_numeric(Nit_unf['Lab2'])

#%%
sns.set_theme(font_scale = 1.25,style='ticks')
for c1 in Lab1_fil.columns:
    for c2 in Lab2_fil.columns:
        
        if c1==c2:
            
            stack = np.concatenate((Lab1_fil[c1],Lab2_fil[c2],Lab1_unf[c1],Lab2_unf[c2]))
            
            slope, intercept, r_value, p_value, std_err =\
                stats.linregress(Lab1_fil[c1],Lab2_fil[c2])
            
            x_reg = np.array([min(stack),max(stack)])
            y_reg = x_reg*slope+intercept
                
            y_text = min(stack)+(max(stack)-min(stack))*0.05
            x_text = max(stack)-(max(stack)-min(stack))*0.4
            
            plt.figure()
            plt.scatter(Lab1_fil[c1],Lab2_fil[c2],s = 50,label = 'Filtered')
            plt.scatter(Lab1_unf[c1],Lab2_unf[c2],facecolor = 'none',edgecolor = 'orange',
                        s = 50,label = 'Unfiltered')
            plt.plot(x_reg,x_reg,'--k',label = '1:1 line')
            plt.plot(x_reg,y_reg,'-k',label = 'regression line')
            plt.xlabel('Lab 1 '+c1+ ' (mg/L)')
            plt.ylabel('Lab 2 '+c1+ ' (mg/L)')
            plt.legend()
            plt.text(x_text,y_text,r'$r^2 =$'+str(np.round(r_value,3))+
                     '\n'+'slope = '+str(np.round(slope,2))+'\n'+'intercept = '+
                     str(np.round(intercept,3)))

#%% make plot for Nitrate < 1
sns.set_theme(font_scale = 1.25,style='ticks')
c = 'Nitrate-N'
c1 = c
c2 = c

Nit1_fil = Lab1_fil[c1][Lab1_fil[c1]<1]
Nit2_fil = Lab2_fil[c2][Lab1_fil[c1]<1]

Nit1_unf = Lab1_unf[c1][Lab1_unf[c1]<1]
Nit2_unf = Lab2_unf[c2][Lab1_unf[c1]<1]

Nit_stack = np.concatenate((Nit1_fil,Nit2_fil,Nit1_unf,Nit2_unf))

slope, intercept, r_value, p_value, std_err =\
    stats.linregress(Nit1_fil,Nit2_fil)
    
x_reg = np.array([min(Nit_stack),max(Nit_stack)])
y_reg = x_reg*slope+intercept
    
y_text = min(Nit_stack)+(max(Nit_stack)-min(Nit_stack))*0.05
x_text = max(Nit_stack)-(max(Nit_stack)-min(Nit_stack))*0.4

plt.figure()
plt.scatter(Nit1_fil,Nit2_fil,s = 50,label = 'Filtered')
plt.scatter(Nit1_unf,Nit2_unf,facecolor = 'none',edgecolor = 'orange',
            s = 50,label = 'Unfiltered')
plt.plot(x_reg,x_reg,'--k',label = '1:1 line')
plt.plot(x_reg,y_reg,'-k',label = 'regression line')
plt.xlabel('Lab 1 '+c1+ ' (mg/L)')
plt.ylabel('Lab 2 '+c1+ ' (mg/L)')
plt.legend()
plt.text(x_text,y_text,r'$r^2 =$'+str(np.round(r_value,3))+
         '\n'+'slope = '+str(np.round(slope,2))+'\n'+'intercept = '+
         str(np.round(intercept,3)))

#%% t-test

for c1 in Lab1_fil.columns:
    for c2 in Lab2_fil.columns:
        
        if c1==c2:
            
            print(c1+' - '+'filtereed')
            pval = stats.ttest_rel(Lab1_fil[c1],Lab2_fil[c2]).pvalue
            print('  p-value: '+str(round(pval,5)))
            print(c1+' - '+'unfiltereed')
            pval = stats.ttest_rel(Lab1_unf[c1],Lab2_unf[c2]).pvalue
            print('  p-value: '+str(round(pval,5)))

#%% t-test for sub Nitrate

c = 'Nitrate-N'
c1 = c
c2 = c

Nit1_fil = Lab1_fil[c1][Lab1_fil[c1]<1]
Nit2_fil = Lab2_fil[c2][Lab1_fil[c1]<1]

Nit1_unf = Lab1_unf[c1][Lab1_unf[c1]<1]
Nit2_unf = Lab2_unf[c2][Lab1_unf[c1]<1]

print('two-sided')
print(c1+' - '+'filtereed')
pval = stats.ttest_rel(Nit1_fil,Nit2_fil).pvalue
print('  p-value: '+str(round(pval,5)))
print(c1+' - '+'unfiltereed')
pval = stats.ttest_rel(Nit1_unf,Nit2_unf).pvalue
print('  p-value: '+str(round(pval,5)))

#%% One-sided t-test

print('----------------')
print('Alternative: Lab 1 > Lab 2')
print(c1+' - '+'filtereed')
pval = stats.ttest_rel(Nit1_fil,Nit2_fil,alternative='greater').pvalue
print('  p-value: '+str(round(pval,5)))
print(c1+' - '+'unfiltereed')
pval = stats.ttest_rel(Nit1_unf,Nit2_unf,alternative='greater').pvalue
print('  p-value: '+str(round(pval,5)))

print('----------------')
print('Alternative: Lab 1 < Lab 2')
print(c1+' - '+'filtereed')
pval = stats.ttest_rel(Nit1_fil,Nit2_fil,alternative='less').pvalue
print('  p-value: '+str(round(pval,5)))
print(c1+' - '+'unfiltereed')
pval = stats.ttest_rel(Nit1_unf,Nit2_unf,alternative='less').pvalue
print('  p-value: '+str(round(pval,5)))

#%%
"""
Anova
"""
df2_col = wq_df2.columns
df2_species = df2_col[1:6]
wq_df2_long = pd.melt(wq_df2,id_vars = ['Name','Date_col','Filtered'],value_vars = df2_species)
wq_df2_long.rename(columns = {'variable':'Species','value':'Conc'},inplace=True)
wq_df2_long['Lab']='Lab2'

wq_df['Lab']='Lab1'
wq_df['Filtered']=True

wq_df_nar = wq_df[wq_df2_long.columns]
