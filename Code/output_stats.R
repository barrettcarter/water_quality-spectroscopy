#### Author: J. Barrett Carter
#### Date: 09/25/2023

#### Description:
#### The purpose of this code is to perform statistical analysis of ML training
#### outputs.

#### Bring in data

proj_dir = 'C:/Users/barre/Documents/GitHub/water_quality-spectroscopy'

sample_type = 'Hydroponics' # used for defining directories

output_dir = paste(proj_dir, sample_type, 'outputs', sep = '/')

output_files = list.files(output_dir)

output_files = output_files[c(17,20,23,24)]

output_files = output_files[c(2,3,4,1)]

outputs_df = read.csv(paste(output_dir,output_files[1], sep = '/'))

for file in output_files[2:4]:
  
  outputs_df = rbind(outputs_df,read.csv(paste(output_dir,file, sep = '/')))

for a in c(1,2,3):
  
  print(a)