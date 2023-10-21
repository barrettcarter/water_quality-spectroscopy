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

proj_dir = 'C:/Users/barre/Documents/GitHub/water_quality-spectroscopy' # For laptop

# proj_dir = 'D:/GitHub/PhD/water_quality-spectroscopy' # for work computer

sample_type = 'Hydroponics' # used for defining directories

output_dir = paste(proj_dir, sample_type, 'outputs', sep = '/')

# figure_dir = 'C:\\Users\\carter_j\\OneDrive\\Research\\PhD\\Communications\\Images\\HNS results' # for work computer

figure_dir = 'C:\\Users\\barre\\OneDrive\\Research\\PhD\\Communications\\Images\\HNS results' # for laptop

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

############ RMSE PLOTS ####################################
test_rmses= subset(outputs_df,output=='test_rmse')

p_rmse = ggplot(test_rmses, aes(x = model, y = value, fill = model)) +
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set1')+
  ylab('test RMSE')+
  labs(title = 'Undiluted HNS')

p_rmse

# ggsave(filename = 'HNS_rmse_boxplot.png', plot = p_rmse, path = figure_dir, 
#        device = 'png', dpi = 300)
  
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
  
  # check to see if sp is not significantly different from any others
  
  if(any(grepl(s,c(dunn_sp_ins_df$comp1,dunn_sp_ins_df$comp2)))){
    
    # make sub dataframes containing all rows with sp
    
    dunn_sp_ins_sub = dunn_sp_ins_df[grepl(s,dunn_sp_ins_df$comparison),]
    
    # dunn_sp_sig_sub = dunn_sp_sig_df[grepl(s,dunn_sp_sig_df$comparison),]
    
    # get list of all species in group and sort
    
    group_s = unique(append(dunn_sp_ins_sub$comp1,dunn_sp_ins_sub$comp2))
    
    group_s = sort(group_s)
    
    # Check if any pair in group_s is significantly different
    
    sig_pairs = subset(dunn_sp_sig_df, comp1 %in% group_s & comp2 %in% group_s)
    
    max_i = nrow(dunn_sp_sig_df)
    
    i = 0
    
    while (nrow(sig_pairs)>0){
      
      sp_sig_all = append(sig_pairs$comp1,sig_pairs$comp2)
      
      sp_sig_unq = unique(sp_sig_all)
      
      sp_sig_cnt = data.frame(sp_sig = sp_sig_unq, count = 0)
      
      for (sp_sig_i in 1:length(sp_sig_unq)){
        
        sp_sig_cnt$count[sp_sig_i]=sum(sp_sig_all==sp_sig_unq[sp_sig_i])
        
      }
      
      sp_drop = sp_sig_cnt$sp_sig[sp_sig_cnt$count==max(sp_sig_cnt$count)]
      
      group_s = group_s[group_s != sp_drop]
      
      sig_pairs = subset(dunn_sp_sig_df, comp1 %in% group_s & comp2 %in% group_s)
      
      i = i + 1
      
      if (i == max_i){
        
        break
        
      }
      
    }
    
    # see if sp group already exists using function
    
    group_exists_fun = function(groups_sublist, group_list = group_s){
      
      return(listNlist(groups_sublist,group_list))
      
    }
    
    group_exists = any(lapply(dunn_grp_sp,group_exists_fun))
    
    # see if species are already grouped together (sub-group)
    
    for (group_sp in names(dunn_grp_sp)){
      
      if (all(group_s %in% dunn_grp_sp[[group_sp]])){
        
        sub_group = TRUE
        
        break
        
      }else{
        
        sub_group = F
        
      }
      
    }
    
    if (length(dunn_grp_sp)==0){
      
      sub_group = F
      
    }
    
    # if group does not exist, create group
    
    if (group_exists==F & sub_group == F){
      
      dunn_grp_sp[[group_letter]]=group_s
      
      ss = group_s[3] #for testing
      
      # also add group letter to every sp in group
      
      for (ss in group_s){
        
        # append if sp already has groups
        
        if (any(grepl(ss,names(dunn_sp_groups)))){
          
          dunn_sp_groups[[ss]] = append(dunn_sp_groups[[ss]],group_letter)  
          
        } else{
          
          # create sp and assign group if sp doesn't already have groups
          
          dunn_sp_groups[[ss]] = group_letter
          
        }
        
      }
      
      # go to next group letter
      
      li = li + 1
      
    }
    
  }else{
    
    # this is for the case that a sp is significantly different from all others
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

ggsave(filename = 'HNS_rsq_dunn-sp_boxplot.png', plot = p_rsq, path = figure_dir,
       device = 'png', dpi = 150, width = 8, height = 8, units = 'in')

### plot with subplots based on groups - no outliers ###

p_rsq_no.outl = ggplot(test_rsqs_grps, aes(x = species, y = value)) +
  geom_boxplot(outlier.shape = NA)+
  facet_wrap(~groups, scale = 'free')+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - post-hoc comparisons - outliers removed')+
  coord_cartesian(ylim = c(-1,1))

p_rsq_no.outl

ggsave(filename = 'HNS_rsq_dunn-sp_boxplot_no-outliers.png', plot = p_rsq_no.outl, path = figure_dir,
       device = 'png', dpi = 150, width = 8, height = 8, units = 'in')


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
  
  # check to see if mod is not significantly different from any others
  
  if(any(grepl(s,c(dunn_mod_ins_df$comp1,dunn_mod_ins_df$comp2)))){
    
    # make sub dataframes containing all rows with mod
    
    dunn_mod_ins_sub = dunn_mod_ins_df[grepl(s,dunn_mod_ins_df$comparison),]
    
    # dunn_mod_sig_sub = dunn_mod_sig_df[grepl(s,dunn_mod_sig_df$comparison),]
    
    # get list of all models in group and sort
    
    group_s = unique(append(dunn_mod_ins_sub$comp1,dunn_mod_ins_sub$comp2))
    
    group_s = sort(group_s)
    
    # Check if any pair in group_s is significantly different
    
    sig_pairs = subset(dunn_mod_sig_df, comp1 %in% group_s & comp2 %in% group_s)
    
    max_i = nrow(dunn_mod_sig_df)
    
    i = 0
    
    while (nrow(sig_pairs)>0){
      
      mod_sig_all = append(sig_pairs$comp1,sig_pairs$comp2)
      
      mod_sig_unq = unique(mod_sig_all)
      
      mod_sig_cnt = data.frame(mod_sig = mod_sig_unq, count = 0)
      
      for (mod_sig_i in 1:length(mod_sig_unq)){
        
        mod_sig_cnt$count[mod_sig_i]=sum(mod_sig_all==mod_sig_unq[mod_sig_i])
        
      }
      
      mod_drop = mod_sig_cnt$mod_sig[mod_sig_cnt$count==max(mod_sig_cnt$count)]
      
      group_s = group_s[group_s != mod_drop]
      
      sig_pairs = subset(dunn_mod_sig_df, comp1 %in% group_s & comp2 %in% group_s)
      
      i = i + 1
      
      if (i == max_i){
        
        break
        
      }
      
    }
    
    # see if mod group already exists using function
    
    group_exists_fun = function(groups_sublist, group_list = group_s){
      
      return(listNlist(groups_sublist,group_list))
      
    }
    
    group_exists = any(lapply(dunn_grp_mod,group_exists_fun))
    
    # see if models are already grouped together (sub-group)
    
    for (group_mod in names(dunn_grp_mod)){
      
      if (all(group_s %in% dunn_grp_mod[[group_mod]])){
        
        sub_group = TRUE
        
        break
        
      }else{
        
        sub_group = F
        
      }
      
    }
    
    if (length(dunn_grp_mod)==0){
      
      sub_group = F
      
    }
    
    # if group does not exist, create group
    
    if (group_exists==F & sub_group == F){
      
      dunn_grp_mod[[group_letter]]=group_s
      
      ss = group_s[3] #for testing
      
      # also add group letter to every mod in group
      
      for (ss in group_s){
        
        # append if mod already has groups
        
        if (any(grepl(ss,names(dunn_mod_groups)))){
          
          dunn_mod_groups[[ss]] = append(dunn_mod_groups[[ss]],group_letter)  
          
        } else{
          
          # create mod and assign group if mod doesn't already have groups
          
          dunn_mod_groups[[ss]] = group_letter
          
        }
        
      }
      
      # go to next group letter
      
      li = li + 1
      
    }
    
  }else{
    
    # this is for the case that a mod is significantly different from all others
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

write.csv(dunn_spmod_df, paste(output_dir,'stats','HNS_r-sq_dunn_spmod-ALL.csv',sep='/'), row.names = F)

### This is where the results are subset so only same-species comparisons are included.
### Uncomment if this is what it wanted for analysis.

dunn_spmod_df = subset(dunn_spmod_df,sp1 == sp2)

write.csv(dunn_spmod_df, paste(output_dir,'stats','HNS_r-sq_dunn_spmod.csv',sep='/'), row.names = F)

### Continue with grouping analysis

dunn_spmod_sig_df = subset(dunn_spmod_df, p < 0.05)

write.csv(dunn_spmod_sig_df, paste(output_dir,'stats','HNS_r-sq_dunn-sig_spmod.csv',sep='/'), row.names = F)

dunn_spmod_ins_df = subset(dunn_spmod_df, p > 0.05)

write.csv(dunn_spmod_ins_df, paste(output_dir,'stats','HNS_r-sq_dunn-ins_spmod.csv',sep='/'), row.names = F)

##### Grouping ##############
dunn_spmod_groups = list()

dunn_grp_spmod = list()

s = spmods[1] # for testing

li = 1 # can be used for testing, but must be set to 1 for official analysis

listNlist = function(list_a,list_b){
  
  return(identical(list_a,list_b))
  
}

# for (sp in unique(dunn_spmod_df$sp1)){
#   
#   dunn_spmod_df_sp = subset(dunn_spmod_df,sp1 == sp)
#   
#   spmods = sort(unique(append(dunn_spmod_df_sp$comp1,dunn_spmod_df_sp$comp2)))
#   
#   li = 1
  
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
      
      max_i = nrow(dunn_spmod_sig_df)
      
      i = 0
      
      while (nrow(sig_pairs)>0){
        
        spmod_sig_all = append(sig_pairs$comp1,sig_pairs$comp2)
        
        spmod_sig_unq = unique(spmod_sig_all)
        
        spmod_sig_cnt = data.frame(spmod_sig = spmod_sig_unq, count = 0)
        
        for (spmod_sig_i in 1:length(spmod_sig_unq)){
          
          spmod_sig_cnt$count[spmod_sig_i]=sum(spmod_sig_all==spmod_sig_unq[spmod_sig_i])
          
        }
        
        spmod_drop = spmod_sig_cnt$spmod_sig[spmod_sig_cnt$count==max(spmod_sig_cnt$count)]
        
        group_s = group_s[group_s != spmod_drop]
        
        sig_pairs = subset(dunn_spmod_sig_df, comp1 %in% group_s & comp2 %in% group_s)
        
        i = i + 1
        
        if (i == max_i){
          
          break
          
        }
        
      }
      
      # see if spmod group already exists using function
      
      group_exists_fun = function(groups_sublist, group_list = group_s){
        
        return(listNlist(groups_sublist,group_list))
        
      }
      
      group_exists = any(lapply(dunn_grp_spmod,group_exists_fun))
      
      # see if spmods are already grouped together (sub-group)
      
      for (group_spmod in names(dunn_grp_spmod)){
        
        if (all(group_s %in% dunn_grp_spmod[[group_spmod]])){
          
          sub_group = TRUE
          
          break
          
        }else{
          
          sub_group = F
          
        }
        
      }
      
      if (length(dunn_grp_spmod)==0){
        
        sub_group = F
        
      }
      
      # if group does not exist, create group
      
      if (group_exists==F & sub_group == F){
        
        dunn_grp_spmod[[group_letter]]=group_s
        
        ss = group_s[3] #for testing
        
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
  
# }

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

dunn_spmod_groups_df$species = unlist(lapply(dunn_spmod_groups_df$spmod, FUN = sp_split_spmod))

dunn_spmod_groups_df$model = unlist(lapply(dunn_spmod_groups_df$spmod, FUN = mod_split_spmod_tk))

### Save group information

write.csv(dunn_spmod_groups_df, paste(output_dir,'stats','HNS_r-sq_dunn_spmod_groups.csv',sep='/'), row.names = F)

test_rsqs_grps = merge(test_rsqs, dunn_spmod_groups_df[c('spmod','groups')],by = 'spmod')

### Make figure

p_rsq = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set1')+
  facet_wrap(~species, scale = 'free')+
  geom_text(data = dunn_spmod_groups_df, aes(x = model, y = 2, label = groups))+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - post-hoc comparisons')

p_rsq

ggsave(filename = 'HNS_rsq_dunn-spmod_boxplot.png', plot = p_rsq, path = figure_dir,
       device = 'png', dpi = 150, width = 12, height = 10, units = 'in')

## outliers removed

p_rsq_no.outl = ggplot(test_rsqs, aes(x = model, y = value, fill = model)) +
  geom_boxplot(outlier.shape = NA)+
  scale_fill_brewer(palette = 'Set1')+
  facet_wrap(~species, scale = 'free')+
  geom_text(data = dunn_spmod_groups_df, aes(x = model, y = 1, label = groups))+
  ylab('test r-sq')+
  labs(title = 'Undiluted HNS - post-hoc comparisons - outliers removed')+
  coord_cartesian(ylim = c(-2,1))

p_rsq_no.outl

ggsave(filename = 'HNS_rsq_dunn-spmod_boxplot_outliers-removed.png', plot = p_rsq_no.outl, path = figure_dir,
       device = 'png', dpi = 150, width = 12, height = 10, units = 'in')

##############################################################################
##### rmse values

## species, in general (all models)

dunn_sp = dunn.test(test_rmses$value,test_rmses$species)

dunn_sp_df = data.frame(comparison = dunn_sp$comparisons, p = dunn_sp$P)

split_comp1 = function(x){
  
  unlist(strsplit(x, ' - '))[1]
  
}

split_comp2 = function(x){
  
  unlist(strsplit(x, ' - '))[2]
  
}

dunn_sp_df$comp1 = unlist(lapply(dunn_sp_df$comparison, FUN = split_comp1))
dunn_sp_df$comp2 = unlist(lapply(dunn_sp_df$comparison, FUN = split_comp2))

write.csv(dunn_sp_df, paste(output_dir,'stats','HNS_rmse_dunn_species.csv',sep='/'), row.names = F)

dunn_sp_sig_df = subset(dunn_sp_df, p < 0.05)

write.csv(dunn_sp_sig_df, paste(output_dir,'stats','HNS_rmse_dunn-sig_species.csv',sep='/'), row.names = F)

dunn_sp_ins_df = subset(dunn_sp_df, p > 0.05)

write.csv(dunn_sp_ins_df, paste(output_dir,'stats','HNS_rmse_dunn-ins_species.csv',sep='/'), row.names = F)

dunn_sp_groups = list()

dunn_grp_sp = list()

s = species[3] # for testing

li = 1 # can be used for testing, but must be set to 1 for official analysis

listNlist = function(list_a,list_b){
  
  return(identical(list_a,list_b))
  
}

for (s in species){
  
  group_letter = letters[li]
  
  # check to see if sp is not significantly different from any others
  
  if(any(grepl(s,c(dunn_sp_ins_df$comp1,dunn_sp_ins_df$comp2)))){
    
    # make sub dataframes containing all rows with sp
    
    dunn_sp_ins_sub = dunn_sp_ins_df[grepl(s,dunn_sp_ins_df$comparison),]
    
    # dunn_sp_sig_sub = dunn_sp_sig_df[grepl(s,dunn_sp_sig_df$comparison),]
    
    # get list of all species in group and sort
    
    group_s = unique(append(dunn_sp_ins_sub$comp1,dunn_sp_ins_sub$comp2))
    
    group_s = sort(group_s)
    
    # Check if any pair in group_s is significantly different
    
    sig_pairs = subset(dunn_sp_sig_df, comp1 %in% group_s & comp2 %in% group_s)
    
    max_i = nrow(dunn_sp_sig_df)
    
    i = 0
    
    while (nrow(sig_pairs)>0){
      
      sp_sig_all = append(sig_pairs$comp1,sig_pairs$comp2)
      
      sp_sig_unq = unique(sp_sig_all)
      
      sp_sig_cnt = data.frame(sp_sig = sp_sig_unq, count = 0)
      
      for (sp_sig_i in 1:length(sp_sig_unq)){
        
        sp_sig_cnt$count[sp_sig_i]=sum(sp_sig_all==sp_sig_unq[sp_sig_i])
        
      }
      
      sp_drop = sp_sig_cnt$sp_sig[sp_sig_cnt$count==max(sp_sig_cnt$count)]
      
      group_s = group_s[group_s != sp_drop]
      
      sig_pairs = subset(dunn_sp_sig_df, comp1 %in% group_s & comp2 %in% group_s)
      
      i = i + 1
      
      if (i == max_i){
        
        break
        
      }
      
    }
    
    # see if sp group already exists using function
    
    group_exists_fun = function(groups_sublist, group_list = group_s){
      
      return(listNlist(groups_sublist,group_list))
      
    }
    
    group_exists = any(lapply(dunn_grp_sp,group_exists_fun))
    
    # see if species are already grouped together (sub-group)
    
    for (group_sp in names(dunn_grp_sp)){
      
      if (all(group_s %in% dunn_grp_sp[[group_sp]])){
        
        sub_group = TRUE
        
        break
        
      }else{
        
        sub_group = F
        
      }
      
    }
    
    if (length(dunn_grp_sp)==0){
      
      sub_group = F
      
    }
    
    # if group does not exist, create group
    
    if (group_exists==F & sub_group == F){
      
      dunn_grp_sp[[group_letter]]=group_s
      
      ss = group_s[3] #for testing
      
      # also add group letter to every sp in group
      
      for (ss in group_s){
        
        # append if sp already has groups
        
        if (any(grepl(ss,names(dunn_sp_groups)))){
          
          dunn_sp_groups[[ss]] = append(dunn_sp_groups[[ss]],group_letter)  
          
        } else{
          
          # create sp and assign group if sp doesn't already have groups
          
          dunn_sp_groups[[ss]] = group_letter
          
        }
        
      }
      
      # go to next group letter
      
      li = li + 1
      
    }
    
  }else{
    
    # this is for the case that a sp is significantly different from all others
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

sp_meds = c(by(test_rmses$value, list(test_rmses$species), median))

dunn_sp_groups_df = dunn_sp_groups_df[order(dunn_sp_groups_df$species),]

dunn_sp_groups_df$median = sp_meds

### Save group information

write.csv(dunn_sp_groups_df, paste(output_dir,'stats','HNS_rmse_dunn_sp_groups.csv',sep='/'), row.names = F)


test_rmses_grps = merge(test_rmses, dunn_sp_groups_df[c('species','groups')],by = 'species')


### plot with subplots based on groups ###

p_rmse = ggplot(test_rmses_grps, aes(x = species, y = value)) +
  geom_boxplot()+
  facet_wrap(~groups, scale = 'free')+
  scale_fill_brewer(palette = 'Set1')+
  # geom_text(data = dunn_sp_groups_df, aes(x = species, y = text_locs, label = groups))+
  ylab('test RMSE (mg/L)')+
  labs(title = 'Undiluted HNS - post-hoc comparisons')

p_rmse

ggsave(filename = 'HNS_rmse_dunn-sp_boxplot.png', plot = p_rmse, path = figure_dir,
       device = 'png', dpi = 150, width = 8, height = 8, units = 'in')

### plot with subplots based on groups - no outliers ###

p_rmse_no.outl = ggplot(test_rmses_grps, aes(x = species, y = value)) +
  geom_boxplot(outlier.shape = NA)+
  facet_wrap(~groups, scale = 'free')+
  ylab('test RMSE (mg/L)')+
  labs(title = 'Undiluted HNS - post-hoc comparisons - outliers removed')+
  coord_cartesian(ylim = c(-1,1))

p_rmse_no.outl

ggsave(filename = 'HNS_rmse_dunn-sp_boxplot_no-outliers.png', plot = p_rmse_no.outl, path = figure_dir,
       device = 'png', dpi = 150, width = 8, height = 8, units = 'in')


###########################################################
### Look at model only ###

models = unique(test_rmses$model)

dunn_mod = dunn.test(test_rmses$value,test_rmses$model)

dunn_mod_df = data.frame(comparison = dunn_mod$comparisons, p = dunn_mod$P)

split_comp1 = function(x){
  
  unlist(strsplit(x, ' - '))[1]
  
}

split_comp2 = function(x){
  
  unlist(strsplit(x, ' - '))[2]
  
}

dunn_mod_df$comp1 = unlist(lapply(dunn_mod_df$comparison, FUN = split_comp1))
dunn_mod_df$comp2 = unlist(lapply(dunn_mod_df$comparison, FUN = split_comp2))

write.csv(dunn_mod_df, paste(output_dir,'stats','HNS_rmse_dunn_model.csv',sep='/'), row.names = F)

dunn_mod_sig_df = subset(dunn_mod_df, p < 0.05)

write.csv(dunn_mod_sig_df, paste(output_dir,'stats','HNS_rmse_dunn-sig_model.csv',sep='/'), row.names = F)

dunn_mod_ins_df = subset(dunn_mod_df, p > 0.05)

write.csv(dunn_mod_ins_df, paste(output_dir,'stats','HNS_rmse_dunn-ins_model.csv',sep='/'), row.names = F)

dunn_mod_groups = list()

dunn_grp_mod = list()

s = models[1] # for testing

li = 1 # can be used for testing, but must be set to 1 for official analysis

listNlist = function(list_a,list_b){
  
  return(identical(list_a,list_b))
  
}

for (s in models){
  
  group_letter = letters[li]
  
  # check to see if mod is not significantly different from any others
  
  if(any(grepl(s,c(dunn_mod_ins_df$comp1,dunn_mod_ins_df$comp2)))){
    
    # make sub dataframes containing all rows with mod
    
    dunn_mod_ins_sub = dunn_mod_ins_df[grepl(s,dunn_mod_ins_df$comparison),]
    
    # dunn_mod_sig_sub = dunn_mod_sig_df[grepl(s,dunn_mod_sig_df$comparison),]
    
    # get list of all models in group and sort
    
    group_s = unique(append(dunn_mod_ins_sub$comp1,dunn_mod_ins_sub$comp2))
    
    group_s = sort(group_s)
    
    # Check if any pair in group_s is significantly different
    
    sig_pairs = subset(dunn_mod_sig_df, comp1 %in% group_s & comp2 %in% group_s)
    
    max_i = nrow(dunn_mod_sig_df)
    
    i = 0
    
    while (nrow(sig_pairs)>0){
      
      mod_sig_all = append(sig_pairs$comp1,sig_pairs$comp2)
      
      mod_sig_unq = unique(mod_sig_all)
      
      mod_sig_cnt = data.frame(mod_sig = mod_sig_unq, count = 0)
      
      for (mod_sig_i in 1:length(mod_sig_unq)){
        
        mod_sig_cnt$count[mod_sig_i]=sum(mod_sig_all==mod_sig_unq[mod_sig_i])
        
      }
      
      mod_drop = mod_sig_cnt$mod_sig[mod_sig_cnt$count==max(mod_sig_cnt$count)]
      
      group_s = group_s[group_s != mod_drop]
      
      sig_pairs = subset(dunn_mod_sig_df, comp1 %in% group_s & comp2 %in% group_s)
      
      i = i + 1
      
      if (i == max_i){
        
        break
        
      }
      
    }
    
    # see if mod group already exists using function
    
    group_exists_fun = function(groups_sublist, group_list = group_s){
      
      return(listNlist(groups_sublist,group_list))
      
    }
    
    group_exists = any(lapply(dunn_grp_mod,group_exists_fun))
    
    # see if models are already grouped together (sub-group)
    
    for (group_mod in names(dunn_grp_mod)){
      
      if (all(group_s %in% dunn_grp_mod[[group_mod]])){
        
        sub_group = TRUE
        
        break
        
      }else{
        
        sub_group = F
        
      }
      
    }
    
    if (length(dunn_grp_mod)==0){
      
      sub_group = F
      
    }
    
    # if group does not exist, create group
    
    if (group_exists==F & sub_group == F){
      
      dunn_grp_mod[[group_letter]]=group_s
      
      ss = group_s[3] #for testing
      
      # also add group letter to every mod in group
      
      for (ss in group_s){
        
        # append if mod already has groups
        
        if (any(grepl(ss,names(dunn_mod_groups)))){
          
          dunn_mod_groups[[ss]] = append(dunn_mod_groups[[ss]],group_letter)  
          
        } else{
          
          # create mod and assign group if mod doesn't already have groups
          
          dunn_mod_groups[[ss]] = group_letter
          
        }
        
      }
      
      # go to next group letter
      
      li = li + 1
      
    }
    
  }else{
    
    # this is for the case that a mod is significantly different from all others
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

mod_meds = c(by(test_rmses$value, list(test_rmses$model), median))

dunn_mod_groups_df = dunn_mod_groups_df[order(dunn_mod_groups_df$model),]

dunn_mod_groups_df$median = mod_meds

### Save group information

write.csv(dunn_mod_groups_df, paste(output_dir,'stats','HNS_rmse_dunn_mod_groups.csv',sep='/'), row.names = F)

test_rmses_grps = merge(test_rmses, dunn_mod_groups_df[c('model','groups')],by = 'model')

### Make figure

p_rmse = ggplot(test_rmses, aes(x = model, y = value, fill = model)) +
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set1')+
  geom_text(data = dunn_mod_groups_df, aes(x = model, y = 10, label = groups))+
  ylab('test RMSE (mg/L)')+
  labs(title = 'Undiluted HNS - post-hoc comparisons')

p_rmse

ggsave(filename = 'HNS_rmse_dunn-mod_boxplot.png', plot = p_rmse, path = figure_dir,
       device = 'png', dpi = 300)

## outliers removed

p_rmse_no.outl = ggplot(test_rmses, aes(x = model, y = value, fill = model)) +
  geom_boxplot(outlier.shape = NA)+
  scale_fill_brewer(palette = 'Set1')+
  geom_text(data = dunn_mod_groups_df, aes(x = model, y = 10, label = groups))+
  ylab('test RMSE (mg/L)')+
  labs(title = 'Undiluted HNS - post-hoc comparisons - outliers removed')+
  coord_cartesian(ylim = c(0,40))

p_rmse_no.outl

ggsave(filename = 'HNS_rmse_dunn-mod_boxplot_no-outliers.png', plot = p_rmse_no.outl, path = figure_dir,
       device = 'png', dpi = 300)


################################################################
### Look at species and model together

test_rmses$spmod = paste(test_rmses$species,test_rmses$model,sep = '_')

spmods = sort(unique(test_rmses$spmod))

### make qqplot

qqplot = ggplot(test_rmses, aes(sample = value, color = factor(model)))+
  facet_wrap(~species, scale = 'free')+
  scale_color_brewer(palette = 'Set1')+
  labs(title = 'Undiluted HNS - QQ Plot')+
  stat_qq()+
  stat_qq_line()

qqplot

# ggsave(filename = 'HNS_rmse_spmod_QQplot.png', plot = qqplot, path = figure_dir,
#        device = 'png', dpi = 150, width = 12, height = 10, units = 'in')

# see if log(rmse) is more normally distributed

test_rmses$logvalue = log(test_rmses$value)

qqplot = ggplot(test_rmses, aes(sample = logvalue, color = factor(model)))+
  facet_wrap(~species, scale = 'free')+
  scale_color_brewer(palette = 'Set1')+
  labs(title = 'Undiluted HNS - log-transform QQ Plot')+
  stat_qq()+
  stat_qq_line()

qqplot

# ggsave(filename = 'HNS_rmse_spmod_logQQplot.png', plot = qqplot, path = figure_dir,
#        device = 'png', dpi = 150, width = 12, height = 10, units = 'in')

test_rmses$model = gsub('-','_',test_rmses$model)

test_rmses$species = gsub('-','_',test_rmses$species)

summary(tukey_spmod_fun <- aov(logvalue ~ species + model + species:model, data = test_rmses))
tukey_spmod = TukeyHSD(tukey_spmod_fun, c("species",'model','species:model'), ordered = TRUE)
# plot(tukey_spmod)

tukey_spmod_mat = tukey_spmod$`species:model`

tukey_spmod_df = data.frame(comparison = rownames(tukey_spmod_mat),p = tukey_spmod_mat[,'p adj'])

dunn_spmod_df = data.frame(comparison = dunn_spmod$comparisons, p = dunn_spmod$P)

split_comp1_tk = function(x){
  
  unlist(strsplit(x, '-'))[1]
  
}

split_comp2_tk = function(x){
  
  unlist(strsplit(x, '-'))[2]
  
}

tukey_spmod_df$comp1 = unlist(lapply(tukey_spmod_df$comparison, FUN = split_comp1_tk))
tukey_spmod_df$comp2 = unlist(lapply(tukey_spmod_df$comparison, FUN = split_comp2_tk))

# reduce results down to only model comparisons for each species

sp_split_spmod_tk = function(x){
  
  return(unlist(strsplit(x,':'))[1])
  
}

mod_split_spmod_tk = function(x){
  
  return(unlist(strsplit(x,':'))[2])
  
}

tukey_spmod_df$sp1 = unlist(lapply(tukey_spmod_df$comp1, FUN = sp_split_spmod_tk))
tukey_spmod_df$sp2 = unlist(lapply(tukey_spmod_df$comp2, FUN = sp_split_spmod_tk))

tukey_spmod_df$mod1 = unlist(lapply(tukey_spmod_df$comp1, FUN = mod_split_spmod_tk))
tukey_spmod_df$mod2 = unlist(lapply(tukey_spmod_df$comp2, FUN = mod_split_spmod_tk))

# THESE HAVE TO BE RAN IN THE RIGHT ORDER

tukey_spmod_df$comp1 = gsub('_','-',tukey_spmod_df$comp1)
tukey_spmod_df$comp2 = gsub('_','-',tukey_spmod_df$comp2)

tukey_spmod_df$comp1 = gsub(':','_',tukey_spmod_df$comp1)
tukey_spmod_df$comp2 = gsub(':','_',tukey_spmod_df$comp2)

tukey_spmod_df$mod1 = gsub('_','-',tukey_spmod_df$mod1)
tukey_spmod_df$mod2 = gsub('_','-',tukey_spmod_df$mod2)

tukey_spmod_df$sp1 = gsub('_','-',tukey_spmod_df$sp1)
tukey_spmod_df$sp2 = gsub('_','-',tukey_spmod_df$sp2)

tukey_spmod_df$comparison = paste(tukey_spmod_df$comp1,tukey_spmod_df$comp2, sep = ' - ')

write.csv(tukey_spmod_df, paste(output_dir,'stats','HNS_rmse_tukey_spmod-ALL.csv',sep='/'), row.names = F)

### This is where the results are subset so only same-species comparisons are included.
### Uncomment if this is what it wanted for analysis.

test_rmses$species = gsub('_','-',test_rmses$species)
test_rmses$model = gsub('_','-',test_rmses$model)

tukey_spmod_df = subset(tukey_spmod_df,sp1 == sp2)

write.csv(tukey_spmod_df, paste(output_dir,'stats','HNS_rmse_tukey_spmod.csv',sep='/'), row.names = F)

### Continue with grouping analysis

tukey_spmod_sig_df = subset(tukey_spmod_df, p < 0.05)

write.csv(tukey_spmod_sig_df, paste(output_dir,'stats','HNS_rmse_tukey-sig_spmod.csv',sep='/'), row.names = F)

tukey_spmod_ins_df = subset(tukey_spmod_df, p > 0.05)

write.csv(tukey_spmod_ins_df, paste(output_dir,'stats','HNS_rmse_tukey-ins_spmod.csv',sep='/'), row.names = F)

##### Grouping ##############
tukey_spmod_groups = list()

tukey_grp_spmod = list()

s = spmods[1] # for testing

li = 1 # can be used for testing, but must be set to 1 for official analysis

listNlist = function(list_a,list_b){
  
  return(identical(list_a,list_b))
  
}

# for (sp in unique(tukey_spmod_df$sp1)){
#   
#   tukey_spmod_df_sp = subset(tukey_spmod_df,sp1 == sp)
#   
#   spmods = sort(unique(append(tukey_spmod_df_sp$comp1,tukey_spmod_df_sp$comp2)))
#   
#   li = 1

for (s in spmods){
  
  group_letter = letters[li]
  
  # check to see if spmod is not significantly different from any others
  
  if(any(grepl(s,c(tukey_spmod_ins_df$comp1,tukey_spmod_ins_df$comp2)))){
    
    # make sub dataframes containing all rows with spmod
    
    tukey_spmod_ins_sub = tukey_spmod_ins_df[grepl(s,tukey_spmod_ins_df$comparison),]
    
    # tukey_spmod_sig_sub = tukey_spmod_sig_df[grepl(s,tukey_spmod_sig_df$comparison),]
    
    # get list of all spmods in group and sort
    
    group_s = unique(append(tukey_spmod_ins_sub$comp1,tukey_spmod_ins_sub$comp2))
    
    group_s = sort(group_s)
    
    # Check if any pair in group_s is significantly different
    
    sig_pairs = subset(tukey_spmod_sig_df, comp1 %in% group_s & comp2 %in% group_s)
    
    max_i = nrow(tukey_spmod_sig_df)
    
    i = 0
    
    while (nrow(sig_pairs)>0){
      
      spmod_sig_all = append(sig_pairs$comp1,sig_pairs$comp2)
      
      spmod_sig_unq = unique(spmod_sig_all)
      
      spmod_sig_cnt = data.frame(spmod_sig = spmod_sig_unq, count = 0)
      
      for (spmod_sig_i in 1:length(spmod_sig_unq)){
        
        spmod_sig_cnt$count[spmod_sig_i]=sum(spmod_sig_all==spmod_sig_unq[spmod_sig_i])
        
      }
      
      spmod_drop = spmod_sig_cnt$spmod_sig[spmod_sig_cnt$count==max(spmod_sig_cnt$count)]
      
      group_s = group_s[group_s != spmod_drop]
      
      sig_pairs = subset(tukey_spmod_sig_df, comp1 %in% group_s & comp2 %in% group_s)
      
      i = i + 1
      
      if (i == max_i){
        
        break
        
      }
      
    }
    
    # see if spmod group already exists using function
    
    group_exists_fun = function(groups_sublist, group_list = group_s){
      
      return(listNlist(groups_sublist,group_list))
      
    }
    
    group_exists = any(lapply(tukey_grp_spmod,group_exists_fun))
    
    # see if spmods are already grouped together (sub-group)
    
    for (group_spmod in names(tukey_grp_spmod)){
      
      if (all(group_s %in% tukey_grp_spmod[[group_spmod]])){
        
        sub_group = TRUE
        
        break
        
      }else{
        
        sub_group = F
        
      }
      
    }
    
    if (length(tukey_grp_spmod)==0){
      
      sub_group = F
      
    }
    
    # if group does not exist, create group
    
    if (group_exists==F & sub_group == F){
      
      tukey_grp_spmod[[group_letter]]=group_s
      
      ss = group_s[3] #for testing
      
      # also add group letter to every spmod in group
      
      for (ss in group_s){
        
        # append if spmod already has groups
        
        if (any(grepl(ss,names(tukey_spmod_groups)))){
          
          tukey_spmod_groups[[ss]] = append(tukey_spmod_groups[[ss]],group_letter)  
          
        } else{
          
          # create spmod and assign group if spmod doesn't already have groups
          
          tukey_spmod_groups[[ss]] = group_letter
          
        }
        
      }
      
      # go to next group letter
      
      li = li + 1
      
    }
    
  }else{
    
    # this is for the case that a spmod is significantly different from all others
    # it is its own group.
    
    tukey_spmod_groups[[s]] = group_letter
    tukey_grp_spmod[[group_letter]]=s
    
    li = li + 1
    
  }
  
}

# }

### make plot showing groups

s = names(tukey_spmod_groups)[1]

gs = tukey_spmod_groups[[s]]

gs = paste0(gs,collapse ='')

tukey_spmod_groups_df = data.frame(spmod = s, groups = gs)

for (s in names(tukey_spmod_groups)[2:length(tukey_spmod_groups)]){
  
  gs = tukey_spmod_groups[[s]]
  
  gs = paste0(gs,collapse ='')
  
  tukey_spmod_groups_df[nrow(tukey_spmod_groups_df)+1,] = c(s,gs)
  
}

spmod_meds = c(by(test_rmses$value, list(test_rmses$spmod), median))

tukey_spmod_groups_df = tukey_spmod_groups_df[order(tukey_spmod_groups_df$spmod),]

tukey_spmod_groups_df$median = spmod_meds

tukey_spmod_groups_df$max = c(by(test_rmses$value, list(test_rmses$spmod), max))

q3_fun = function(x){
  
  return(quantile(x,0.75))
  
}

q1_fun = function(x){
  
  return(quantile(x,0.25))
  
}

tukey_spmod_groups_df$q3 = c(by(test_rmses$value, list(test_rmses$spmod), q3_fun))

tukey_spmod_groups_df$q1 = c(by(test_rmses$value, list(test_rmses$spmod), q1_fun))

tukey_spmod_groups_df$iqr = tukey_spmod_groups_df$q3 - tukey_spmod_groups_df$q1
  
tukey_spmod_groups_df$in_upr = tukey_spmod_groups_df$q3 + 1.5*tukey_spmod_groups_df$iqr

# stats by species

sp_med = c(by(test_rmses$value, list(test_rmses$species), median))

sp_med_df = data.frame(species = names(sp_med), sp_med = sp_med)

tukey_spmod_groups_df = merge(tukey_spmod_groups_df,sp_med_df,by.x = 'species',
                              by.y = 'species',all.x = T)

sp_q3 = c(by(test_rmses$value, list(test_rmses$species), q3_fun))

sp_q1 = c(by(test_rmses$value, list(test_rmses$species), q1_fun))

sp_q3_df = data.frame(species = names(sp_q3), sp_q3 = sp_q3)

tukey_spmod_groups_df = merge(tukey_spmod_groups_df,sp_q3_df,by.x = 'species',
                              by.y = 'species',all.x = T)

sp_q1_df = data.frame(species = names(sp_q1), sp_q1 = sp_q1)

tukey_spmod_groups_df = merge(tukey_spmod_groups_df,sp_q1_df,by.x = 'species',
                              by.y = 'species',all.x = T)

tukey_spmod_groups_df$sp_iqr = tukey_spmod_groups_df$sp_q3 - tukey_spmod_groups_df$sp_q1

tukey_spmod_groups_df$sp_in_upr = tukey_spmod_groups_df$sp_q3 + 1.5*tukey_spmod_groups_df$sp_iqr

sp_max = c(by(test_rmses$value, list(test_rmses$species), max))

sp_max_df = data.frame(species = names(sp_max), sp_max = sp_max)

tukey_spmod_groups_df = merge(tukey_spmod_groups_df,sp_max_df,by.x = 'species',
                              by.y = 'species',all.x = T)

sp_in_upr = tukey_spmod_groups_df$sp_q3 + 1.5*tukey_spmod_groups_df$sp_iqr

sp_in_lwr = tukey_spmod_groups_df$sp_q1 - 1.5*tukey_spmod_groups_df$sp_iqr

sp_split_spmod = function(x){
  
  return(unlist(strsplit(x,'_'))[1])
  
}

mod_split_spmod = function(x){
  
  return(unlist(strsplit(x,'_'))[2])
  
}

tukey_spmod_groups_df$species = unlist(lapply(tukey_spmod_groups_df$spmod, FUN = sp_split_spmod))

tukey_spmod_groups_df$model = unlist(lapply(tukey_spmod_groups_df$spmod, FUN = mod_split_spmod))

### Save group information

write.csv(tukey_spmod_groups_df, paste(output_dir,'stats','HNS_rmse_tukey_spmod_groups.csv',sep='/'), row.names = F)

test_rmses_grps = merge(test_rmses, tukey_spmod_groups_df[c('spmod','groups')],by = 'spmod')

### Make figure

p_rmse = ggplot(test_rmses, aes(x = model, y = value, fill = model)) +
  geom_boxplot()+
  scale_fill_brewer(palette = 'Set1')+
  facet_wrap(~species, scale = 'free')+
  geom_text(data = tukey_spmod_groups_df, aes(x = model, y = in_upr, label = groups))+
  ylab('test RMSE (mg/L)')+
  labs(title = 'Undiluted HNS - post-hoc comparisons')

p_rmse

ggsave(filename = 'HNS_rmse_tukey-spmod_boxplot.png', plot = p_rmse, path = figure_dir,
       device = 'png', dpi = 150, width = 12, height = 10, units = 'in')

## outliers removed

p_rmse_no.outl = ggplot(subset(test_rmses,value<in_upr & value>in_lwr), aes(x = model, y = value, fill = model)) +
  geom_boxplot(outlier.shape = NA)+
  scale_fill_brewer(palette = 'Set1')+
  facet_wrap(~species, scale = 'free')+
  geom_text(data = tukey_spmod_groups_df, aes(x = model, y = sp_in_upr*1.1, label = groups))+
  ylab('test RMSE (mg/L)')+
  labs(title = 'Undiluted HNS - post-hoc comparisons - outliers removed')
  coord_cartesian(ylim = c(sp_in_lwr,sp_in_upr*1.2))

p_rmse_no.outl

ggsave(filename = 'HNS_rmse_tukey-spmod_boxplot_outliers-removed.png', plot = p_rmse_no.outl, path = figure_dir,
       device = 'png', dpi = 150, width = 12, height = 10, units = 'in')
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
