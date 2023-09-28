#### Author: J. Barrett Carter
#### Date: 09/25/2023

#### Description:
#### The purpose of this code is to perform statistical analysis of ML training
#### outputs.

### Import libraries

library(ggplot2)

#### Bring in data

# proj_dir = 'C:/Users/barre/Documents/GitHub/water_quality-spectroscopy' # For laptop

proj_dir = 'D:/GitHub/PhD/water_quality-spectroscopy' # for work computer

sample_type = 'Hydroponics' # used for defining directories

output_dir = paste(proj_dir, sample_type, 'outputs', sep = '/')

figure_dir = 'C:\\Users\\carter_j\\OneDrive\\Research\\PhD\\Communications\\Images\\HNS results'

output_files = as.vector(list.files(output_dir))

output_files = output_files[c(17,20,23,24)]

output_files = output_files[c(2,3,4,1)]

file = output_files[1]

outputs_df = read.csv(paste(output_dir,file, sep = '/'))

outputs_df$sample_type = unlist(strsplit(file,'_'))[1]

outputs_df$model = unlist(strsplit(file,'_'))[2]

for (file in output_files[2:4]){
  
  output_df = read.csv(paste(output_dir,file, sep = '/'))
  
  output_df$sample_type = unlist(strsplit(file,'_'))[1]
  
  output_df$model = unlist(strsplit(file,'_'))[2]
  
  outputs_df = rbind(outputs_df,output_df)
  
}

outputs = unique(outputs_df$output)

test_rsqs = subset(outputs_df,output=='test_rsq')

p_rsq = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS')
  
p_rsq

ggsave(filename = 'HNS_rsq_boxplot.png', plot = p_rsq, path = figure_dir, 
       device = 'png', dpi = 300)

### plot with removed outliers


p_rsq_no.outl = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot(outlier.shape = NA)+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - outliers removed')+
  coord_cartesian(ylim = c(-1,1))
  
p_rsq_no.outl

ggsave(filename = 'HNS_rsq_boxplot_no-outliers.png', plot = p_rsq_no.outl, path = figure_dir, 
       device = 'png', dpi = 300)

#### separated by species

p_rsq_sp = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot()+
  facet_wrap(~species, scale = 'free')+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS')
  
p_rsq_sp

ggsave(filename = 'HNS_rsq-v-species_boxplot.png', plot = p_rsq_sp, path = figure_dir, 
       device = 'png', dpi = 150, width = 12, height = 10, units = 'in')

### plot with removed outliers

p_rsq_no.outl = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot(outlier.shape = NA)+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - outliers removed')+
  coord_cartesian(ylim = c(-1,1))

p_rsq_no.outl

ggsave(filename = 'HNS_rsq_boxplot_no-outliers.png', plot = p_rsq_no.outl, path = figure_dir, 
       device = 'png', dpi = 300)

oneway.test(value~model, data = test_rsqs)


test_rmses= subset(outputs_df,output=='test_rmse')


