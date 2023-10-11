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
  labs(title ='Undiluted HNS')

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

write.csv(dunn_sp_df, paste(output_dir,'stats','HNS_r-sq_dunn_species.csv',sep='/'), row.names = F)

dunn_sp_sig_df = subset(dunn_sp_df, p < 0.05)

write.csv(dunn_sp_sig_df, paste(output_dir,'stats','HNS_r-sq_dunn-sig_species.csv',sep='/'), row.names = F)

dunn_sp_ins_df = subset(dunn_sp_df, p > 0.05)

write.csv(dunn_sp_ins_df, paste(output_dir,'stats','HNS_r-sq_dunn-ins_species.csv',sep='/'), row.names = F)

dunn_sp_groups = list()

dunn_grp_sp = list()

s = species[3] # for testing

li = 1 # can be used for testing, but must be set to 1 for official analysis

listNlist = function(list_a,list_b){
  
  return(identical(list_a,list_b))
  
}

for (s in species){
  
  group_letter = letters[li]
  
  # check to see if species is not significantly different from any others
  
  if(any(grepl(s,c(dunn_sp_ins_df$comp1,dunn_sp_ins_df$comp2)))){
    
    # make sub dataframe containing all rows with species
    
    dunn_sp_ins_sub = dunn_sp_ins_df[grepl(s,dunn_sp_ins_df$comparison),]
    
    # get list of all species in group and sort
    
    group_s = unique(append(dunn_sp_ins_sub$comp1,dunn_sp_ins_sub$comp2))
    
    group_s = sort(group_s)
    
    # see if species group already exists using function
    
    group_exists_fun = function(groups_sublist, group_list = group_s){
      
      return(listNlist(groups_sublist,group_list))
      
    }
    
    group_exists = any(lapply(dunn_grp_sp,group_exists_fun))
    
    # if group does not exist, create group
    
    if (group_exists==F){
      
      dunn_grp_sp[[group_letter]]=group_s
      
      ss = group_s[1] #for testing
      
      # also add group letter to every species in group
      
      for (ss in group_s){
        
        # append if species already has groups
        
        if (any(grepl(ss,names(dunn_sp_groups)))){
          
          dunn_sp_groups[[ss]] = append(dunn_sp_groups[[ss]],group_letter)  
          
        } else{
          
          # create species and assign group if species doesn't already have groups
          
          dunn_sp_groups[[ss]] = group_letter
          
        }
        
      }
      
      # go to next group letter
      
      li = li + 1
      
    }
    
  }else{
    
    # this is for the case that a species is significantly different from all others
    # it is its own group.
    
    dunn_sp_groups[[s]] = group_letter
    dunn_grp_sp[[group_letter]]=s
    
    li = li + 1
    
  }
  
}

### make plot showing groups

s = names(dunn_sp_groups)[1]

gs = dunn_sp_groups[[s]]

gs = paste0(gs,collapse ='')

dunn_sp_groups_df = data.frame(species = s, groups = gs)

for (s in names(dunn_sp_groups)[2:length(dunn_sp_groups)]){
  
  gs = dunn_sp_groups[[s]]
  
  gs = paste0(gs,collapse ='')
  
  dunn_sp_groups_df[nrow(dunn_sp_groups_df)+1,] = c(s,gs)
  
}

sp_meds = c(by(test_rsqs$value, list(test_rsqs$species), median))

dunn_sp_groups_df = dunn_sp_groups_df[order(dunn_sp_groups_df$species),]

dunn_sp_groups_df$median = sp_meds

### Save group information

write.csv(dunn_sp_groups_df, paste(output_dir,'stats','HNS_r-sq_dunn_sp_groups.csv',sep='/'), row.names = F)


test_rsqs_grps = merge(test_rsqs, dunn_sp_groups_df[c('species','groups')],by = 'species')


### plot with subplots based on groups ###

p_rsq = ggplot(test_rsqs_grps, aes(x = species, y = value)) +
  geom_boxplot()+
  facet_wrap(~groups, scale = 'free')+
  scale_fill_brewer(palette = 'Set1')+
  # geom_text(data = dunn_sp_groups_df, aes(x = species, y = text_locs, label = groups))+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - post-hoc comparisons')

p_rsq

# ggsave(filename = 'HNS_rsq_dunn-sp_boxplot.png', plot = p_rsq, path = figure_dir,
#        device = 'png', dpi = 300)

### plot with subplots based on groups - no outliers ###

p_rsq_no.outl = ggplot(test_rsqs_grps, aes(x = species, y = value)) +
  geom_boxplot(outlier.shape = NA)+
  facet_wrap(~groups, scale = 'free')+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - post-hoc comparisons - outliers removed')+
  coord_cartesian(ylim = c(-1,1))

p_rsq_no.outl

# ggsave(filename = 'HNS_rsq_dunn-sp_boxplot_no-outliers.png', plot = p_rsq_no.outl, path = figure_dir,
#        device = 'png', dpi = 300)


###########################################################
### Look at model only ###

models = unique(test_rsqs$model)

dunn_mod = dunn.test(test_rsqs$value,test_rsqs$model)

dunn_mod_df = data.frame(comparison = dunn_mod$comparisons, p = dunn_mod$P)

split_comp1 = function(x){
  
  unlist(strsplit(x, ' - '))[1]
  
}

split_comp2 = function(x){
  
  unlist(strsplit(x, ' - '))[2]
  
}

dunn_mod_df$comp1 = unlist(lapply(dunn_mod_df$comparison, FUN = split_comp1))
dunn_mod_df$comp2 = unlist(lapply(dunn_mod_df$comparison, FUN = split_comp2))

write.csv(dunn_mod_df, paste(output_dir,'stats','HNS_r-sq_dunn_model.csv',sep='/'), row.names = F)

dunn_mod_sig_df = subset(dunn_mod_df, p < 0.05)

write.csv(dunn_mod_sig_df, paste(output_dir,'stats','HNS_r-sq_dunn-sig_model.csv',sep='/'), row.names = F)

dunn_mod_ins_df = subset(dunn_mod_df, p > 0.05)

write.csv(dunn_mod_ins_df, paste(output_dir,'stats','HNS_r-sq_dunn-ins_model.csv',sep='/'), row.names = F)

dunn_mod_groups = list()

dunn_grp_mod = list()

s = models[1] # for testing

li = 1 # can be used for testing, but must be set to 1 for official analysis

listNlist = function(list_a,list_b){
  
  return(identical(list_a,list_b))
  
}

for (s in models){
  
  group_letter = letters[li]
  
  # check to see if model is not significantly different from any others
  
  if(any(grepl(s,c(dunn_mod_ins_df$comp1,dunn_mod_ins_df$comp2)))){
    
    # make sub dataframe containing all rows with model
    
    dunn_mod_ins_sub = dunn_mod_ins_df[grepl(s,dunn_mod_ins_df$comparison),]
    
    # get list of all models in group and sort
    
    group_s = unique(append(dunn_mod_ins_sub$comp1,dunn_mod_ins_sub$comp2))
    
    group_s = sort(group_s)
    
    # see if model group already exists using function
    
    group_exists_fun = function(groups_sublist, group_list = group_s){
      
      return(listNlist(groups_sublist,group_list))
      
    }
    
    group_exists = any(lapply(dunn_grp_mod,group_exists_fun))
    
    # if group does not exist, create group
    
    if (group_exists==F){
      
      dunn_grp_mod[[group_letter]]=group_s
      
      ss = group_s[1] #for testing
      
      # also add group letter to every model in group
      
      for (ss in group_s){
        
        # append if model already has groups
        
        if (any(grepl(ss,names(dunn_mod_groups)))){
          
          dunn_mod_groups[[ss]] = append(dunn_mod_groups[[ss]],group_letter)  
          
        } else{
          
          # create model and assign group if model doesn't already have groups
          
          dunn_mod_groups[[ss]] = group_letter
          
        }
        
      }
      
      # go to next group letter
      
      li = li + 1
      
    }
    
  }else{
    
    # this is for the case that a model is significantly different from all others
    # it is its own group.
    
    dunn_mod_groups[[s]] = group_letter
    dunn_grp_mod[[group_letter]]=s
    
    li = li + 1
    
  }
  
}

### make plot showing groups

bp_txt_fun = function(x){
  
  
  # return(quantile(x, 0.62)) # based on percentile
  
  return(max(x)*2) # multiplier of the max
  
}

s = names(dunn_mod_groups)[1]

gs = dunn_mod_groups[[s]]

gs = paste0(gs,collapse ='')

dunn_mod_groups_df = data.frame(model = s, groups = gs)

for (s in names(dunn_mod_groups)[2:length(dunn_mod_groups)]){
  
  gs = dunn_mod_groups[[s]]
  
  gs = paste0(gs,collapse ='')
  
  dunn_mod_groups_df[nrow(dunn_mod_groups_df)+1,] = c(s,gs)
  
}

mod_meds = c(by(test_rsqs$value, list(test_rsqs$model), median))

dunn_mod_groups_df = dunn_mod_groups_df[order(dunn_mod_groups_df$model),]

dunn_mod_groups_df$median = mod_meds

### Save group information

write.csv(dunn_mod_groups_df, paste(output_dir,'stats','HNS_r-sq_dunn_mod_groups.csv',sep='/'), row.names = F)

test_rsqs_grps = merge(test_rsqs, dunn_mod_groups_df[c('model','groups')],by = 'model')

### Make figure

p_rsq = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set1')+
  geom_text(data = dunn_mod_groups_df, aes(x = model, y = 2, label = groups))+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - post-hoc comparisons')

p_rsq

# ggsave(filename = 'HNS_rsq_dunn-mod_boxplot.png', plot = p_rsq, path = figure_dir,
#        device = 'png', dpi = 300)

## outliers removed

p_rsq_no.outl = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot(outlier.shape = NA)+
  scale_fill_brewer(palette = 'Set1')+
  geom_text(data = dunn_mod_groups_df, aes(x = model, y = 1, label = groups))+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - post-hoc comparisons - outliers removed')+
  coord_cartesian(ylim = c(-1,1))

p_rsq_no.outl

# ggsave(filename = 'HNS_rsq_dunn-mod_boxplot_no-outliers.png', plot = p_rsq_no.outl, path = figure_dir,
#        device = 'png', dpi = 300)


################################################################
### Look at species and model together

test_rsqs$spmod = paste(test_rsqs$species,test_rsqs$model,sep = '_')

spmods = sort(unique(test_rsqs$spmod))

dunn_spmod = dunn.test(test_rsqs$value,test_rsqs$spmod)

dunn_spmod_df = data.frame(comparison = dunn_spmod$comparisons, p = dunn_spmod$P)

split_comp1 = function(x){
  
  unlist(strsplit(x, ' - '))[1]
  
}

split_comp2 = function(x){
  
  unlist(strsplit(x, ' - '))[2]
  
}

dunn_spmod_df$comp1 = unlist(lapply(dunn_spmod_df$comparison, FUN = split_comp1))
dunn_spmod_df$comp2 = unlist(lapply(dunn_spmod_df$comparison, FUN = split_comp2))

# reduce results down to only model comparisons for each species

sp_split_spmod = function(x){
  
  return(unlist(strsplit(x,'_'))[1])
  
}

mod_split_spmod = function(x){
  
  return(unlist(strsplit(x,'_'))[2])
  
}

dunn_spmod_df$sp1 = unlist(lapply(dunn_spmod_df$comp1, FUN = sp_split_spmod))
dunn_spmod_df$sp2 = unlist(lapply(dunn_spmod_df$comp2, FUN = sp_split_spmod))

dunn_spmod_df$mod1 = unlist(lapply(dunn_spmod_df$comp1, FUN = mod_split_spmod))
dunn_spmod_df$mod2 = unlist(lapply(dunn_spmod_df$comp2, FUN = mod_split_spmod))

dunn_spmod_df = subset(dunn_spmod_df,sp1 == sp2)

write.csv(dunn_spmod_df, paste(output_dir,'stats','HNS_r-sq_dunn_spmod.csv',sep='/'), row.names = F)

dunn_spmod_sig_df = subset(dunn_spmod_df, p < 0.05)

write.csv(dunn_spmod_sig_df, paste(output_dir,'stats','HNS_r-sq_dunn-sig_spmod.csv',sep='/'), row.names = F)

dunn_spmod_ins_df = subset(dunn_spmod_df, p > 0.05)

write.csv(dunn_spmod_ins_df, paste(output_dir,'stats','HNS_r-sq_dunn-ins_spmod.csv',sep='/'), row.names = F)



dunn_spmod_groups = list()

dunn_grp_spmod = list()

s = spmods[34] # for testing

li = 1 # can be used for testing, but must be set to 1 for official analysis

listNlist = function(list_a,list_b){
  
  return(identical(list_a,list_b))
  
}

for (s in spmods){
  
  group_letter = letters[li]
  
  # check to see if spmod is not significantly different from any others
  
  if(any(grepl(s,c(dunn_spmod_ins_df$comp1,dunn_spmod_ins_df$comp2)))){
    
    # make sub dataframes containing all rows with spmod
    
    dunn_spmod_ins_sub = dunn_spmod_ins_df[grepl(s,dunn_spmod_ins_df$comparison),]
    
    # dunn_spmod_sig_sub = dunn_spmod_sig_df[grepl(s,dunn_spmod_sig_df$comparison),]
    
    # get list of all spmods in group and sort
    
    group_s = unique(append(dunn_spmod_ins_sub$comp1,dunn_spmod_ins_sub$comp2))
    
    group_s = sort(group_s)
    
    # Check if any pair in group_s is significantly different
    
    sig_pairs = subset(dunn_spmod_sig_df, comp1 %in% group_s & comp2 %in% group_s)
    
    spmod_drop = unique(append(sig_pairs$comp1,sig_pairs$comp2))
    
    # see if spmod group already exists using function
    
    group_exists_fun = function(groups_sublist, group_list = group_s){
      
      return(listNlist(groups_sublist,group_list))
      
    }
    
    group_exists = any(lapply(dunn_grp_spmod,group_exists_fun))
    
    # if group does not exist, create group
    
    if (group_exists==F){
      
      dunn_grp_spmod[[group_letter]]=group_s
      
      ss = group_s[1] #for testing
      
      # also add group letter to every spmod in group
      
      for (ss in group_s){
        
        # append if spmod already has groups
        
        if (any(grepl(ss,names(dunn_spmod_groups)))){
          
          dunn_spmod_groups[[ss]] = append(dunn_spmod_groups[[ss]],group_letter)  
          
        } else{
          
          # create spmod and assign group if spmod doesn't already have groups
          
          dunn_spmod_groups[[ss]] = group_letter
          
        }
        
      }
      
      # go to next group letter
      
      li = li + 1
      
    }
    
  }else{
    
    # this is for the case that a spmod is significantly different from all others
    # it is its own group.
    
    dunn_spmod_groups[[s]] = group_letter
    dunn_grp_spmod[[group_letter]]=s
    
    li = li + 1
    
  }
  
}

### make plot showing groups

s = names(dunn_spmod_groups)[1]

gs = dunn_spmod_groups[[s]]

gs = paste0(gs,collapse ='')

dunn_spmod_groups_df = data.frame(spmod = s, groups = gs)

for (s in names(dunn_spmod_groups)[2:length(dunn_spmod_groups)]){
  
  gs = dunn_spmod_groups[[s]]
  
  gs = paste0(gs,collapse ='')
  
  dunn_spmod_groups_df[nrow(dunn_spmod_groups_df)+1,] = c(s,gs)
  
}

spmod_meds = c(by(test_rsqs$value, list(test_rsqs$spmod), median))

dunn_spmod_groups_df = dunn_spmod_groups_df[order(dunn_spmod_groups_df$spmod),]

dunn_spmod_groups_df$median = spmod_meds

### Save group information

write.csv(dunn_spmod_groups_df, paste(output_dir,'stats','HNS_r-sq_dunn_spmod_groups.csv',sep='/'), row.names = F)

test_rsqs_grps = merge(test_rsqs, dunn_spmod_groups_df[c('spmod','groups')],by = 'spmod')

### Make figure

p_rsq = ggplot(test_rsqs, aes(x = spmod, y = value, fill = spmod)) +
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set1')+
  geom_text(data = dunn_spmod_groups_df, aes(x = spmod, y = 2, label = groups))+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - post-hoc comparisons')

p_rsq

# ggsave(filename = 'HNS_rsq_dunn-spmod_boxplot.png', plot = p_rsq, path = figure_dir,
#        device = 'png', dpi = 300)

## outliers removed

p_rsq_no.outl = ggplot(test_rsqs, aes(x = spmod, y = value, fill = spmod)) +
  geom_boxplot(outlier.shape = NA)+
  scale_fill_brewer(palette = 'Set1')+
  geom_text(data = dunn_spmod_groups_df, aes(x = spmod, y = 1, label = groups))+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - post-hoc comparisons - outliers removed')+
  coord_cartesian(ylim = c(-1,1))

p_rsq_no.outl

# ggsave(filename = 'HNS_rsq_dunn-spmod_boxplot_no-outliers.png', plot = p_rsq_no.outl, path = figure_dir,
#        device = 'png', dpi = 300)
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
