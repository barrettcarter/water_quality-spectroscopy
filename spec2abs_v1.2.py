# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:25:33 2020

@author: jbarrett.carter
"""

# This code produces an absorbance dataframe from the spectral dataframe

import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import seaborn as sns
# IMPORTANT: Don't have the baseDir and saveDir be the same
user = os.getlogin() 
spec_dir='C:/Users/'+user+'/OneDrive/Documents/spectra/'
spec_df_dir='C:/Users/'+user+'/OneDrive/Documents/spec_df/'
abs_df_dir='C:/Users/'+user+'/OneDrive/Documents/abs/'

spec_df_fn='spectra_df.csv'

# Bring in spectra
spec_df=pd.read_csv(spec_df_dir+spec_df_fn)
# Only use 0deg spectra
spec_df=spec_df[spec_df.Meas_ang=='0deg']

# Add Datetime of analysis column (Datetime_an)
spec_df['Datetime_an'] = spec_df.Date_an+' '+spec_df.Time_an
spec_df.Datetime_an = pd.to_datetime(spec_df.Datetime_an)
# spec_df.Datetime_an =\
# spec_df.Datetime_an.apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f'))

### fix collection date formats
## the 7-16-20 dates can't be converted to datetime
## they also all need to be the same format

bad_dates = spec_df.Date_col == '7-16-20'
spec_df.loc[bad_dates,'Date_col']='7-16-2020'

def move_yr_back(date_string):
    if date_string.split('-')[0]==str(2020):
        new_date = date_string.split('-')[1]+'-'+date_string.split('-')[2]+\
        '-'+date_string.split('-')[0]
        return(new_date)
    else:
        return(date_string)

spec_df.Date_col =\
spec_df.Date_col.apply(lambda x: move_yr_back(x))

# now convert to timestamp

# spec_df.Date_col = pd.to_datetime(spec_df.Date_col) # may not be necessary

# spec_df.Date_col =\
# spec_df.Date_col.apply(lambda x: dt.datetime.strptime(x,'%m-%d-%Y'))

# spec_df.Date_col=spec_df.Date_col.astype('str') #to convert back to string

# Separate into references and filtered and unfiltered samples
refs_df = spec_df.loc[spec_df.Name=='ref'].reset_index(drop = True)
samp_df = spec_df.loc[spec_df.Name!='ref'].reset_index(drop = True)

ref_ints_full = refs_df.loc[:,'band_1':'band_1024']
ref_max_inds = ref_ints_full.idxmax(axis=1)

ref_ints_200 = refs_df.loc[:,'band_1':'band_200']
ref_max_inds_200 = ref_ints_200.idxmax(axis=1)
ref_ints_full = ref_ints_full.to_numpy()

# One of the samples is mislabeled and needs to be dropped
samp_df.drop(284,inplace=True)
samp_df.reset_index(drop=True,inplace=True)
sam_ints_full = samp_df.loc[:,'band_1':'band_1024']
sam_max_inds = sam_ints_full.idxmax(axis=1)
sam_ints_full = sam_ints_full.to_numpy()

# Based on the indices of maximum values, band_741 seems most common (col 740)
# Based on previous analysis, band_170 is good indicator of reference quality

b741s = sam_ints_full[:,740] #samples
b741r = ref_ints_full[:,740] # references
b170r = ref_ints_full[:,169] # references

# plt.figure()
# plt.hist(b741s, label = 'band_741 - samples')
# plt.legend()

# plt.figure()
# plt.hist(b741r, label = 'band_741 - references')
# plt.legend()

# plt.figure()
# plt.hist(b170r, label = 'band_170 - references')
# plt.legend()

# b741s_stats = stats.describe(b741s)
# b741s_sd = np.sqrt(b741s_stats[3])

# b741r_stats = stats.describe(b741r)
# b741r_sd = np.sqrt(b741r_stats[3])

# keep_s = np.logical_and(b741s > 11000,b741s < 13000)
# keep_r = np.logical_and(b741r > 11000,b741r < 13000)
keep_s = np.logical_and(b741s > 10000,b741s < 14500)
keep_r = np.logical_and(b741r > 11000,b741r < 14500)
keep_r = np.logical_and(keep_r,b170r > 4500)

samp_df = samp_df.loc[keep_s].reset_index(drop = True)
refs_df = refs_df.loc[keep_r].reset_index(drop = True)

ref_ints_full = refs_df.loc[:,'band_1':'band_1024']
ref_ints_full = ref_ints_full.to_numpy()

sam_ints_full = samp_df.loc[:,'band_1':'band_1024']
sam_ints_full = sam_ints_full.to_numpy()

b741s = sam_ints_full[:,740] #samples
b741r = ref_ints_full[:,740] # references

plt.figure()
plt.hist(b741s, label = 'band_741 - samples(screened)')
plt.legend()

plt.figure()
plt.hist(b741r, label = 'band_741 - references(screened)')
plt.legend()

### Pair samples and references based on analysis times
### THIS DID NOT WORK WELL. BASED ON SPECTRA MAY WORK BETTER
## worked better after filtering

time_dif_mat = np.empty([refs_df.shape[0],samp_df.shape[0]])
for n in range(time_dif_mat.shape[1]):
    time_dif_mat[:,n]=samp_df.Datetime_an[n]-refs_df.Datetime_an

time_dif_mat = time_dif_mat/10**15
time_dif_mat = abs(time_dif_mat)

mins = np.amin(time_dif_mat,axis=0) # min value in each column
# np.where(time_dif_mat==mins[50])[1][0]
# create vectors to store indices describing sample-reference pairs
samp_inds = np.empty([samp_df.shape[0]])
ref_inds = np.empty([samp_df.shape[0]])

for i in range(len(samp_inds)):
    samp_inds[i]=np.where(time_dif_mat==mins[i])[1][0]
    ref_inds[i]=np.where(time_dif_mat==mins[i])[0][0]
    
# ### Pair samples and references based on difference between spectra maximums
# ### Alternative to pairing based on time

# spec_dif_mat = np.empty([refs_df.shape[0],samp_df.shape[0]])
# for n in range(spec_dif_mat.shape[1]):
#     spec_dif_mat[:,n]=samp_df.band_741[n]-refs_df.band_741

# spec_dif_mat = abs(spec_dif_mat)

# mins = np.amin(spec_dif_mat,axis=0) # min value in each column
# # np.where(time_dif_mat==mins[50])[1][0]
# # create vectors to store indices describing sample-reference pairs
# samp_inds = np.empty([samp_df.shape[0]])
# ref_inds = np.empty([samp_df.shape[0]])

# # There seems to be a problem caused by difference values of 0

# plt.figure()
# plt.plot(sam_ints_full_a[79],label = 's79')
# plt.plot(ref_ints_full[39],label = 'r39')
# plt.legend()

# for i in range(len(samp_inds)):
#     samp_inds[i]=np.where(spec_dif_mat==mins[i])[1][0]
#     ref_inds[i]=np.where(spec_dif_mat==mins[i])[0][0]

# isolate reference-sample combinations
refs = refs_df.iloc[ref_inds].reset_index(drop = True)
sams = samp_df.iloc[samp_inds].reset_index(drop = True)
sam_ids = sams.iloc[:,0:3].to_numpy()
sam_ids_pd = sams.iloc[:,0]+sams.iloc[:,1].apply(lambda x: str(x))+\
    sams.iloc[:,2].apply(lambda x: str(x))

# Analyzing the isolated references and samples
# ref_ints = refs.loc[:,'band_1':'band_1024']
# ref_max_inds = ref_ints.idxmax(axis=1)

# ref_ints_200 = refs.loc[:,'band_1':'band_200']
# ref_max_inds_200 = ref_ints_200.idxmax(axis=1)

# ref_ints = ref_ints.to_numpy()

# # it looks like band 170 is a good indicator of reference quality

# b170r = ref_ints[:,169] # references

# plt.figure()
# plt.hist(b170r, label = 'band_170 - references(screened)')
# plt.legend()

# it look like 4000 is a good cutoff of band_170


# Not sure if I should do this step
# # only keep data points where int_times match
# keep = sams.int_Time==refs.int_Time
# refs = refs.loc[keep].reset_index(drop = True)
# sams = sams.loc[keep].reset_index(drop = True)

abs_df = sams

ref_ints = refs.loc[:,'band_1':'band_1024']
sam_ints = sams.loc[:,'band_1':'band_1024']

ref_ints = ref_ints.to_numpy()
sam_ints = sam_ints.to_numpy()

ratio = ref_ints/sam_ints
ratio = np.asarray(ratio).astype(np.float64)

abs_ints=np.log10(ratio)

abs_df.loc[:,'band_1':'band_1024']=abs_ints

max_abs = np.amax(abs_ints,axis=1)
max_abs_sr = pd.Series(max_abs)


### some visualizations and stats and more screening

# max_abs_sr.describe()

# plt.figure()
# plt.boxplot(max_abs)
# plt.figure()
# plt.hist(max_abs)

# q1 = max_abs_sr.quantile(q=0.25)
# q3 = max_abs_sr.quantile(q=0.75)
# iqr = q3-q1
# min_co=q1-1.5*iqr
# max_co=q3+1.5*iqr

# plt.figure()
# for i in range(refs.shape[0]):
#     plt.plot(abs_ints[i])

# for i in range(refs.shape[0]):
#     plt.figure()    
#     plt.plot(abs_ints[i],label = abs_df.iloc[i,0:3])
#     plt.legend(loc='upper right')

# plt.figure()
# plt.subplot(211)    
# plt.plot(abs_ints[0],label = abs_df.iloc[0,0:3])
# plt.legend(loc='upper right')
# plt.subplot(212)
# plt.plot(ref_ints.loc[0,:],label =refs.loc[0,'Datetime_an'])
# plt.plot(sam_ints.loc[0,:],label = sams.loc[0,'Name'])
# plt.legend(loc='upper left')

# plt.figure()
# plt.subplot(211)    
# plt.plot(abs_ints[1],label = abs_df.iloc[[1],0:3])
# plt.legend(loc='upper right')
# plt.subplot(212)
# plt.plot(ref_ints[1],label =refs.loc[1,'Datetime_an'])
# plt.plot(sam_ints[1],label = sams.loc[1,'Name'])
# plt.legend(loc='upper left')

# for i in range(refs.shape[0]):
#     plt.figure()
#     plt.subplot(211)    
#     plt.plot(abs_ints[i],label = abs_df.iloc[i,0:3].values)
#     plt.legend(loc='upper right')
#     plt.subplot(212)
#     plt.plot(ref_ints[i],label =refs.loc[i,'Datetime_an'])
#     plt.plot(sam_ints[i],label = str(i)+'-'+sams.loc[i,'Name'])
#     plt.legend(loc='upper left')
    
# Looking at the graphs, the following spectra (by row number) appear to be erroneous
# Bad references (may be good sample spectrum)
# 40, 41, 63, 71, 78, 83, 91, 92, 94, 95, 96, 98,101, 108, 113, 131, 141, 161,
# 162, 171, 173, 174, 175, 177, 
# Can be deleted (bad sample spectrum or has good replicate):
# 48, 100, 108

# abs_df.drop([48,100,108],inplace=True)
# # abs_df.reset_index(drop=True,inplace=True)
# abs_ints = abs_df.loc[:,'band_1':'band_1024']
# bands = np.linspace(184.2, 667.6,num = 1024)
# bands = np.around(bands,2)
# abs_ints.columns=bands
    
# for i in range(2):
#     plt.figure()
#     plt.subplot(211)    
#     plt.plot(abs_ints[i],label = abs_df.iloc[i,0:3].values)
#     plt.legend(loc='upper right')
#     plt.subplot(212)
#     plt.plot(ref_ints[i],label =refs.loc[i,'Datetime_an'])
#     plt.plot(sam_ints[i],label = sams.loc[i,'Name'])
#     plt.legend(loc='upper left')
    
# ref_a = ref_ints[0]
# sam_a = sam_ints[0]
# abs_a = np.log10(ref_a/sam_a)
# plt.figure()
# plt.title(sam_ids[0,0]+str(sam_ids[0,1])+sam_ids[0,2])
# plt.plot(abs_a)

sam_ids_un = sam_ids_pd.unique()
reps = sam_ids_pd.shape[0]/sam_ids_un.shape[0] #approximation of average number of replicates

### Plotting spectra and references for inspections

def spec_plots(reference_array,sample_array,sample_id):
    abs_array = np.log10(reference_array/sample_array)
    plt.figure()
    plt.title(sample_id)
    plt.subplot(211)    
    plt.plot(abs_array,label = 'absorbance')
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(reference_array,label ='reference')
    plt.plot(sample_array,label = sample_id)
    plt.legend(loc='upper left')
    
def ref_date(row_number):
    date = abs_df.loc[48,'Date_an']
    refs_date = refs_df.loc[refs_df.Date_an==date,'Date_an']
    return(refs_date)

ref_date(48)
refs.loc[48,'Date_an']

################################
# making a heatmap

sns.heatmap(abs_ints)
sns.heatmap(ref_ints)

### save dataframe as csv
# to convert timestamps to strings
abs_df.Datetime_an=abs_df.Datetime_an.astype('str')

abs_df.to_csv(abs_df_dir+'abs_df.csv')

# the rest of this code is not valid, but may be useful
##############################################################3

i = 178
spec_plots(ref_ints[i],sam_ints[i],str(refs.iloc[[i],:].index[0])+'-'+
           sam_ids[i,0]+str(sam_ids[i,1])+sam_ids[i,2])

r = 152
plt.figure()
plt.plot(ref_ints[r],label = r)
plt.legend(loc = 'upper right')

r=281
spec_plots(ref_ints_full[r],sam_ints[i],str(sams.iloc[[i],:].index[0])+'-'+
           sam_ids[i,0]+str(sam_ids[i,1])+sam_ids[i,2])

# row 1 has bad reading
sams.drop(1,inplace=True)
refs.drop(1,inplace=True)

ref_ints = refs.loc[:,'band_1':'band_1024']
sam_ints = sams.loc[:,'band_1':'band_1024']

ref_ints = ref_ints.to_numpy()
sam_ints = sam_ints.to_numpy()

sam_ids = sams.iloc[:,0:3].to_numpy()

# row 3 has bad reading
sams.drop(3,inplace=True)
refs.drop(3,inplace=True)

ref_ints = refs.loc[:,'band_1':'band_1024']
sam_ints = sams.loc[:,'band_1':'band_1024']

ref_ints = ref_ints.to_numpy()
sam_ints = sam_ints.to_numpy()

sam_ids = sams.iloc[:,0:3].to_numpy()

# row 6 has problem with reference
refs.loc[6,:]=refs_df.loc[257,:]
ref_ints = refs.loc[:,'band_1':'band_1024']
ref_ints = ref_ints.to_numpy()

# row 8 has problem with ref
refs.loc[8,:]=refs_df.loc[561,:]
ref_ints = refs.loc[:,'band_1':'band_1024']
ref_ints = ref_ints.to_numpy()

# row 9 has problem with ref
refs.loc[9,:]=refs_df.loc[561,:]
ref_ints = refs.loc[:,'band_1':'band_1024']
ref_ints = ref_ints.to_numpy()

# row 10 has problem with ref
refs.loc[10,:]=refs_df.loc[561,:]
ref_ints = refs.loc[:,'band_1':'band_1024']
ref_ints = ref_ints.to_numpy()

### ROWS 11 - 14 MAY BE OK. NEED TO CHECK FILTERED SAMPLES
# row 11 and 12 have bad samples

sams.drop([11,12],inplace=True)
refs.drop([11,12],inplace=True)

ref_ints = refs.loc[:,'band_1':'band_1024']
sam_ints = sams.loc[:,'band_1':'band_1024']

ref_ints = ref_ints.to_numpy()
sam_ints = sam_ints.to_numpy()

sam_ids = sams.iloc[:,0:3].to_numpy()

# row 13 and 14 have bad samples

sams.drop([13,14],inplace=True)
refs.drop([13,14],inplace=True)

ref_ints = refs.loc[:,'band_1':'band_1024']
sam_ints = sams.loc[:,'band_1':'band_1024']

ref_ints = ref_ints.to_numpy()
sam_ints = sam_ints.to_numpy()

sam_ids = sams.iloc[:,0:3].to_numpy()