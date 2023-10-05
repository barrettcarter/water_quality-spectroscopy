#### Author: J. Barrett Carter
#### Date: 09/25/2023

#### Description:
#### The purpose of this code is to perform statistical analysis of ML training
#### outputs.

### Import libraries

library(ggplot2)
library(tidyr)
library(dunn.test)

#### Bring in data

# proj_dir = 'C:/Users/barre/Documents/GitHub/water_quality-spectroscopy' # For laptop

proj_dir = 'D:/GitHub/PhD/water_quality-spectroscopy' # for work computer

sample_type = 'Hydroponics' # used for defining directories

output_dir = paste(proj_dir, sample_type, 'outputs', sep = '/')

figure_dir = 'C:\\Users\\carter_j\\OneDrive\\Research\\PhD\\Communications\\Images\\HNS results' # for work computer

# figure_dir = 'C:\\Users\\barre\\OneDrive\\Research\\PhD\\Communications\\Images\\HNS results' # for laptop

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

species = unique(outputs_df$species)

########### R-SQUARED PLOTS ####################

test_rsqs = subset(outputs_df,output=='test_rsq')

p_rsq = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS')
  
p_rsq

# ggsave(filename = 'HNS_rsq_boxplot.png', plot = p_rsq, path = figure_dir, 
#        device = 'png', dpi = 300)

### plot with removed outliers


p_rsq_no.outl = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot(outlier.shape = NA)+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - outliers removed')+
  coord_cartesian(ylim = c(-1,1))
  
p_rsq_no.outl

# ggsave(filename = 'HNS_rsq_boxplot_no-outliers.png', plot = p_rsq_no.outl, path = figure_dir, 
#        device = 'png', dpi = 300)

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

p_rsq_sp_no.outl = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot(outlier.shape = NA)+
  facet_wrap(~species, scale = 'free')+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - outliers removed')+
  coord_cartesian(ylim = c(-1,1))

p_rsq_sp_no.outl

ggsave(filename = 'HNS_rsq-v-species_boxplot_no-outliers.png', plot = p_rsq_sp_no.outl, path = figure_dir, 
       device = 'png', dpi = 150, width = 12, height = 10, units = 'in')


############ RMSE PLOTS ####################################
test_rmses= subset(outputs_df,output=='test_rmse')

p_rmse = ggplot(test_rmses, aes(x = model, y = value, fill = model)) +
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test RMSE')+
  labs(title = 'Undiluted HNS')

p_rmse

ggsave(filename = 'HNS_rmse_boxplot.png', plot = p_rmse, path = figure_dir, 
       device = 'png', dpi = 300)

### plot with removed outliers


p_rmse_no.outl = ggplot(test_rmses, aes(x = model, y = value, fill = model)) +
  geom_boxplot(outlier.shape = NA)+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test RMSE')+
  labs(title = 'Undiluted HNS - outliers removed')+
  coord_cartesian(ylim = c(0,50))

p_rmse_no.outl

ggsave(filename = 'HNS_rmse_boxplot_no-outliers.png', plot = p_rmse_no.outl, path = figure_dir, 
       device = 'png', dpi = 300)

#### separated by species

p_rmse_sp = ggplot(test_rmses, aes(x = model, y = value, fill = model)) +
  geom_boxplot()+
  facet_wrap(~species, scale = 'free')+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test RMSE')+
  labs(title = 'Undiluted HNS')

p_rmse_sp

ggsave(filename = 'HNS_rmse-v-species_boxplot.png', plot = p_rmse_sp, path = figure_dir, 
       device = 'png', dpi = 150, width = 12, height = 10, units = 'in')

### plot with removed outliers

p_rmse_sp_no.outl = ggplot(test_rmses, aes(x = model, y = value, fill = model)) +
  geom_boxplot(outlier.shape = NA)+
  facet_wrap(~species, scale = 'free')+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test RMSE')+
  labs(title = 'Undiluted HNS - outliers removed')+
  coord_cartesian(ylim = c(0,50))

p_rmse_sp_no.outl

ggsave(filename = 'HNS_rmse-v-species_boxplot_no-outliers.png', plot = p_rmse_sp_no.outl, path = figure_dir, 
       device = 'png', dpi = 150, width = 12, height = 10, units = 'in')

############# 1:1 Plots #####################

ys = subset(outputs_df, output == 'y_true_test'|output=='y_hat_test')

ys = ys[order(ys$output, decreasing = T),]

test_inds = subset(outputs_df, output == 'test_ind')

test_inds = rbind(test_inds,test_inds)

ys$index = test_inds$value

y_hats = subset(ys,output == 'y_hat_test')

y_trues = subset(ys,output == 'y_true_test')

ys_wide = pivot_wider(ys,names_from = output, values_from = value)

m = 'PLS'

for (m in unique(ys_wide$model)){
  
  p_ys = ggplot(subset(ys_wide,model == m), aes(x = y_true_test, y = y_hat_test)) +
    geom_point(color = 'grey')+
    stat_smooth(method = 'lm', formula = y~x, geom = 'smooth')+
    geom_abline(slope = 1, intercept = 0, linetype = 'dashed',size = 1)+
    facet_wrap(~species, scale = 'free')+
    # scale_fill_brewer(palette = 'Set1')+
    ylab('Predicted Concentration (mg/L)')+
    xlab('True Concentration (mg/L)')+
    labs(title = paste0('Undiluted HNS - ',m))
  
  p_ys
  
  ggsave(filename = paste0('HNS_fitplot_',m,'.png'),
         plot = p_ys, path = figure_dir,
         device = 'png', dpi = 150, width = 10, height = 7.5, units = 'in')
  
}

### plot showing all models

p_ys_m = ggplot(ys_wide, aes(x = y_true_test, y = y_hat_test, col = model)) +
  # geom_point(alpha = 0.05)+
  stat_smooth(method = 'lm', formula = y~x, geom = 'smooth')+
  geom_abline(slope = 1, intercept = 0, linetype = 'dashed',size = 1, color = 'black')+
  facet_wrap(~species, scale = 'free')+
  scale_color_brewer(palette = 'Set1')+
  ylab('Predicted Concentration (mg/L)')+
  xlab('True Concentration (mg/L)')+
  labs(title = paste0('Undiluted HNS - ',m))

p_ys_m

ggsave(filename = paste0('HNS_fitplot_mods.png'),
       plot = p_ys_m, path = figure_dir,
       device = 'png', dpi = 150, width = 10, height = 7.5, units = 'in')
  
################## STATISTICS ########################################

##### r-sq values

## species, in general (all models)

dunn_sp = dunn.test(test_rsqs$value,test_rsqs$species)

dunn_sp_df = data.frame(comparison = dunn_sp$comparisons, p = dunn_sp$P)

split_comp1 = function(x){
  
  unlist(strsplit(x, ' - '))[1]
  
}

split_comp2 = function(x){
  
  unlist(strsplit(x, ' - '))[2]
  
}

dunn_sp_df$comp1 = unlist(lapply(dunn_sp_df$comparison, FUN = split_comp1))
dunn_sp_df$comp2 = unlist(lapply(dunn_sp_df$comparison, FUN = split_comp2))

write.csv(dunn_sp_df, paste(output_dir,'HNS_r-sq_dunn_species.csv',sep='/'), row.names = F)

dunn_sp_sig_df = subset(dunn_sp_df, p < 0.05)

write.csv(dunn_sp_sig_df, paste(output_dir,'HNS_r-sq_dunn-sig_species.csv',sep='/'), row.names = F)

dunn_sp_ins_df = subset(dunn_sp_df, p > 0.05)

write.csv(dunn_sp_ins_df, paste(output_dir,'HNS_r-sq_dunn-ins_species.csv',sep='/'), row.names = F)

dunn_sp_groups = list()

for (s in species){
  
  dunn_sp_groups = append(dunn_sp_groups,list(s = c('z','z')))
  
}

names(dunn_sp_groups) = species

dunn_grp_sp = list()

s = species[3] # for testing

li = 1 # can be used for testing, but must be set to 1 for official analysis

listNlist = function(list_a,list_b){
  
  return(identical(list_a,list_b))
  
}

for (s in species){
  
  group_letter = letters[li]
  
  dunn_sp_sig_sub = dunn_sp_sig_df[grepl(s,dunn_sp_ins_df$comparison),]
  
  group_s = unique(append(dunn_sp_sig_sub$comp1,dunn_sp_sig_sub$comp2))
  
  group_s = sort(group_s)
  
  if (length(dunn_grp_sp)==0){
    
    dunn_grp_sp[[group_letter]]=group_s
    
  }
  
  group_exists_fun = function(groups_sublist, group_list = group_s){
    
    return(listNlist(groups_sublist,group_list))
    
  }
  
  group_exists = any(lapply(dunn_grp_sp,group_exists_fun))
  
  if (group_exists==F){
    
    dunn_grp_sp[[group_letter]]=group_s
    
    for (ss in group_s){
      
      if (any(grepl(ss,names(dunn_sp_groups)))){
        
        dunn_sp_groups[[ss]] = append(dunn_sp_groups[[ss]],group_letter)  
        
      } else{
        
        dunn_sp_groups[[ss]] = group_letter
        
      }
    
    }
    
    li = li + 1
    
  }
  
}




#################################################################
###     Scratch Code                              


dunn_sp_groups_unique = unique(dunn_sp_groups)

dunn_sp_groups = append(dunn_sp_groups,list(A = c('a','b')))

dunn_sp_groups$A = append(dunn_sp_groups$A,'c')

dunn_sp_groups = append(dunn_sp_groups,list(B = c('d','e')))

foo <- list(a = 1, b = list(c = "a", d = FALSE))
bar <- modifyList(foo, list(e = 2, b = list(d = TRUE)))
str(foo)
str(bar)
