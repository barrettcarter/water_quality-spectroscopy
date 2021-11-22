# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 08:50:22 2021

@author: jbarrett.carter
"""

import pandas as pd

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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

#%%
################################################################################ DEFAULTS

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
output_dir = os.path.join(path_to_wqs,'Streams/outputs/')
results_pls_fn = 'abs_wq_df_streams.csv'
spectra_path = os.path.join(spectra_path,abs_wq_fn)
os.path.exists(spectra_path)
np.random.seed(7)