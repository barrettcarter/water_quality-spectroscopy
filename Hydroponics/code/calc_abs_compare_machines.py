#%% Bring in libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#%% Define functions
################################################# FUNCTIONS
def calc_abs(sample_array,reference_array,baseline_corr = 0):

    # if type(wavelengths) == str:
    #     wavelengths = np.linspace(1,len(sample_array))

    # if baseline_corr == 0:
    #     absorbance = np.log10(reference_array/sample_array)
        
    # else:
        
    #     sample_array = sample_array - baseline_corr*min(np.concatenate((reference_array,sample_array)))
    #     reference_array = reference_array - baseline_corr*min(np.concatenate((reference_array,sample_array)))
    #     absorbance = np.log10(reference_array/sample_array)
    
    sample_array = sample_array - baseline_corr*min(np.concatenate((reference_array,sample_array)))
    reference_array = reference_array - baseline_corr*min(np.concatenate((reference_array,sample_array)))
    absorbance = np.log10(reference_array/sample_array)

    return absorbance
#%% Set paths and bring in data

user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
abs_df_dir=os.path.join(path_to_wqs,'Hydroponics/inputs/absorbance/')
specDir=os.path.join(path_to_wqs,'Hydroponics/inputs/spectra/') 

abs_df_fn = 'abs_HNSs_nanodrop_all.csv'
spectrum_fn = 'spec_HNSs46_False_01-13-2022_0deg_20700.0_UFABE_2022-01-19T17%3A53%3A47.634943.csv'
ref_fn = 'spec_ref_0deg_20700.0_UFABE_2022-01-19T17%3A01%3A34.011511.csv'

# Bring in data
nd_abs_df=pd.read_csv(abs_df_dir+abs_df_fn)
gs_spec = pd.read_csv(specDir+spectrum_fn)
ref_spec = pd.read_csv(specDir+ref_fn)

#%%
############################### Bring in files
gs_wavelengths = ref_spec.WVL.to_numpy()
nd_wavelengths = nd_abs_df['Wavelength (nm)'].to_numpy()
ref_ints = ref_spec.INT.to_numpy()
nd_abs = nd_abs_df[' HNSs46'].to_numpy()
gs_ints = gs_spec.INT.to_numpy()
#%%
### Calculate absorbances
blc = 0.999
gs_abs = calc_abs(gs_ints,ref_ints)
gs_abs_corr = calc_abs(gs_ints,ref_ints,baseline_corr=blc)

#%% make plots
plt.figure()
plt.plot(gs_wavelengths[0:1000],gs_abs[0:1000],label = 'gs')
plt.plot(nd_wavelengths[0:600],nd_abs[0:600],label = 'nd')
plt.title('original (uncorrected)')
plt.xlabel('wavelength (nm)')
plt.ylabel('absorbance')
plt.legend(loc='upper right')

plt.figure()
plt.plot(gs_wavelengths[0:1000],gs_abs_corr[0:1000],label = 'gs')
plt.plot(nd_wavelengths[0:600],nd_abs[0:600],label = 'nd')
#plt.ylim(0,3)
plt.title('baseline corrected ('+str(blc)+')')
plt.xlabel('wavelength (nm)')
plt.ylabel('absorbance')
plt.legend(loc='upper right')



plt.figure()
plt.plot(gs_wavelengths,gs_ints,label = 'HNSs46')
plt.plot(gs_wavelengths,ref_ints,label = 'reference')
plt.title('spectra')
plt.xlabel('wavelength (nm)')
plt.ylabel('power (counts)')
plt.legend(loc='upper left')
plt.ylim([0,16000])

#%%


