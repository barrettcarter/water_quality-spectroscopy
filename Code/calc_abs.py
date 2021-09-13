#%%
print('Loading modules')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
print('...done loading modules')
#%%
################################################# FUNCTIONS
def calc_abs(sample_array,reference_array,baseline_corr = False, 
             wavelengths = 'missing'):

    if type(wavelengths) == str:
        wavelengths = np.linspace(1,len(sample_array))

    if baseline_corr == False:
        absorbance = np.log10(reference_array/sample_array)
        
    else:
        
        sample_array = sample_array - 0.95*min(reference_array)
        reference_array = reference_array - 0.95*min(reference_array)
        absorbance = np.log10(reference_array/sample_array)

    return absorbance
#%%
################################################# DEFAULTS
user = os.getlogin()
specDir='C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data\\spectra\\'      # LOCATION WHERE SPECTRA ARE SAVED
#%%
############################### Bring in files

refName = 'spec_ref_0deg_20700.0_UFABE_2021-09-02T13%3A19%3A30.163411.csv'          # reference spectrum
refSpec_df = pd.read_csv(specDir+refName)
waveLengths = pd.DataFrame.to_numpy(refSpec_df.WVL)

ref = refSpec_df.INT
ref = ref.to_numpy()

sample1Name = 'spec_HNSs1d30_False_08-25-2021_0deg_20700.0_UFABE_2021-09-02T13%3A25%3A09.795173.csv'
sample2Name = 'spec_HNSs2d30_False_08-30-2021_0deg_20700.0_UFABE_2021-09-02T13%3A31%3A05.583398.csv'

sample1 = pd.read_csv(specDir+sample1Name)
sample1 = sample1.INT
sample1 = sample1.to_numpy()
sample2 = pd.read_csv(specDir+sample2Name)
sample2 = sample2.INT
sample2 = sample2.to_numpy()


#%%
### Calculate absorbances

abs_1 = calc_abs(sample1,ref,wavelengths = waveLengths)
abs_2 = calc_abs(sample2,ref,wavelengths = waveLengths)

abs_1_corr = calc_abs(sample1,ref,wavelengths = waveLengths,baseline_corr=True)
abs_2_corr = calc_abs(sample2,ref,wavelengths = waveLengths,baseline_corr=True)
#%%

### make plots
plt.figure()
plt.plot(waveLengths[0:300],abs_1[0:300],label = 'HNSs1d30')
plt.plot(waveLengths[0:300],abs_2[0:300],label = 'HNSs2d30')
plt.title('original (uncorrected)')
plt.xlabel('wavelength (nm)')
plt.ylabel('absorbance')
plt.legend(loc='upper right')

plt.figure()
plt.plot(waveLengths[0:300],abs_1_corr[0:300],label = 'HNSs1d30')
plt.plot(waveLengths[0:300],abs_2_corr[0:300],label = 'HNSs2d30')
plt.ylim(0,3)
plt.title('baseline corrected')
plt.xlabel('wavelength (nm)')
plt.ylabel('absorbance')
plt.legend(loc='upper right')



plt.figure()
plt.plot(waveLengths,sample1,label = 'HNSs1d30')
plt.plot(waveLengths,sample2,label = 'HNSs2d30')
plt.plot(waveLengths,refSpec_df.INT,label = 'reference')
plt.title('spectra')
plt.xlabel('wavelength (nm)')
plt.ylabel('power (counts)')
plt.legend(loc='upper left')
plt.ylim([0,16000])

#%%


