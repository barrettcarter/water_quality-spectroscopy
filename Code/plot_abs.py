print('Loading modules')
import pandas as pd
import matplotlib.pyplot as plt
import os

print('...done loading modules')
#%%
user = os.getlogin() 
path_to_wqs = 'C:\\Users\\'+user+'\\OneDrive\\Research\\PhD\\Data_analysis\\water_quality-spectroscopy\\'
abs_df_dir=os.path.join(path_to_wqs,'Data/spectra/')

abs_df_fn = 'abs_df_u2d.csv'

# Bring in data
abs_df=pd.read_csv(abs_df_dir+abs_df_fn)
#%%

############################### Bring in files

specName = 'spec_hat_filtered_09-24-2020ref_90deg_22500.0_UFABE_2020-10-13T12%3A25%3A54.732857.csv'          # reference spectrum
Spec_df = pd.read_csv(specDir+specName)
waveLengths = pd.DataFrame.to_numpy(Spec_df.WVL)
counts = pd.DataFrame.to_numpy(Spec_df.INT)

plt.figure()
plt.plot(waveLengths,counts)
plt.title('specName')
plt.xlabel('wavelength (nm)')
plt.ylabel('count')
# plt.show() # use this when in VS Code
plt.show(block = False) # use this when out of VS Code