#%%
print('Loading modules')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
print('...done loading modules')
#%%
################################################# FUNCTIONS
def calc_abs(sampleFile,referenceFile,specDirectory):

    global refSpec_array  
    global sampleSpec_array

    sampleName = sampleFile
    refName = referenceFile

    sampleSpec_df = pd.read_csv(specDirectory+sampleName)
    refSpec_df = pd.read_csv(specDirectory+refName)

    sampleSpec_array = pd.DataFrame.to_numpy(sampleSpec_df.INT)
    refSpec_array = pd.DataFrame.to_numpy(refSpec_df.INT)
   
    absorbance = np.log10(refSpec_array/sampleSpec_array)

    return absorbance
#%%
################################################# DEFAULTS
user = os.getlogin()
specDir='C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data\\spectra\\'      # LOCATION WHERE SPECTRA ARE SAVED
#%%
############################### Bring in files

refName = 'spec_ref_0deg_22500.0_UFABE_2021-01-08T10%3A15%3A35.160053.csv'          # reference spectrum
refSpec_df = pd.read_csv(specDir+refName)
waveLengths = pd.DataFrame.to_numpy(refSpec_df.WVL)

sample1Name = 'spec_hat_unfiltered_01-05-2021_0deg_22500.0_UFABE_2021-01-08T10%3A16%3A08.936184.csv'
sample2Name = 'spec_posnw16_unfiltered_01-05-2021_0deg_22500.0_UFABE_2021-01-08T10%3A30%3A12.475933.csv'

sample1 = pd.read_csv(specDir+sample1Name)

sample2 = pd.read_csv(specDir+sample2Name)
#%%
### Calculate absorbances

spec1 = calc_abs(sample1Name,refName,specDir)
spec2 = calc_abs(sample2Name,refName,specDir)
#%%

### make plots
plt.figure()
plt.plot(waveLengths,spec1,label = 'hat unfiltered')
plt.plot(waveLengths,spec2,label = 'posnw16 unfiltered')
plt.title('otow lysimeter samples')
plt.xlabel('wavelength (nm)')
plt.ylabel('absorbance')
plt.legend(loc='upper right')


plt.figure()
plt.plot(waveLengths,sample1.INT,label = 'hat unfiltered')
plt.plot(waveLengths,sample2.INT,label = 'posnw16 unfiltered')
plt.plot(waveLengths,refSpec_df.INT,label = 'reference')
plt.title('spectra')
plt.xlabel('wavelength (nm)')
plt.ylabel('power (counts)')
plt.legend(loc='upper left')
plt.ylim([0,16000])

#%%


