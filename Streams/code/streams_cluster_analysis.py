# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:09:45 2021

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
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.utils import resample
# from sklearn.metrics import mean_squared_error as MSE
# from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

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

#%% Get absorbance data

abs_df = abs_wq_df.loc[:,'band_1':]

#%% Do PCA

pca = PCA(n_components = 2)
abs_red = pca.fit_transform(abs_df)

#%% make PCA plot

plt.figure()
plt.scatter(abs_red[:,0],abs_red[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')

#%% do clustering

clusters = np.empty([abs_df.shape[0],3])

for clusts in [2,3,4]:

    clustering = AgglomerativeClustering(n_clusters = clusts).fit(abs_df)
    clusters[:,clusts-2] = clustering.labels_
    
    plt.figure()
    plt.scatter(abs_red[:,0],abs_red[:,1],c = clustering.labels_,cmap = 'viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
#%% make plots in 3D

pca3 = PCA(n_components = 3)
abs_red3 = pca3.fit_transform(abs_df)

for clusts in [0,1,2]:
    
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter3D(abs_red3[:,0],abs_red3[:,1],abs_red3[:,2],c = clusters[:,clusts],cmap = 'viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    ax.set_zlabel('PC3')

#%% make new dataframe with clusters

# abs_wq_clust_df = abs_wq_df.copy()
# abs_wq_clust_df['2-Cluster-Cat']=clusters[:,0]
# abs_wq_clust_df['3-Cluster-Cat']=clusters[:,1]
# abs_wq_clust_df['4-Cluster-Cat']=clusters[:,2]

#%% insert columns in dataframe

abs_wq_df.drop(labels = 'Unnamed: 0',axis=1,inplace=True)
abs_wq_df.insert(0,'2-Cluster-Cat',clusters[:,0])
abs_wq_df.insert(1,'3-Cluster-Cat',clusters[:,1])
abs_wq_df.insert(2,'4-Cluster-Cat',clusters[:,2])
#%% save dataframe

abs_wq_df.to_csv(inter_dir+'abs_wq_df_streams.csv',index=False)
