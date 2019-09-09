library(tidyverse)
library(pbapply)
library(parallel)
library(assertthat)
library(pROC)
library(patchwork)
source('R/utils.R')

theme_set(theme_minimal() + 
            theme(panel.grid.minor = element_blank()))

species <- list.files('out', '_ssnet.csv', full.names = TRUE) %>%
  gsub('out/', '', .) %>%
  gsub('\\_ssnet.csv', '', .)

load_ll(species[1])

pboptions(use_lb = TRUE)
cl <- makeCluster(parallel::detectCores())
clusterExport(cl, c('load_ll'))
clusterEvalQ(cl, library(assertthat))
clusterEvalQ(cl, library(tidyverse))
clusterEvalQ(cl, source('R/utils.R'))
ll_dfs <- pblapply(species, load_ll, cl = cl)
stopCluster(cl)


ll_df <- ll_dfs %>%
  lapply(function(x) x$nll) %>%
  bind_rows %>%
  filter(group %in% c('train', 'validation')) %>%
  mutate(group = ifelse(group == 'train', 
                        'Training data', 
                        'Validation data'))

ll_df %>%
  write_csv('out/train-valid-nll.csv')

# check for NA values in the NLL values (which result from underflow)
ll_df %>%
  group_by(group) %>%
  summarize(nn_na = sum(nn %>% is.na),
            ss_na = sum(ss %>% is.na), 
            sn_na = sum(ssnn %>% is.na))


overall_comparisons <- ll_df %>%
  group_by(group) %>%
  summarize(nn_nll = mean(nn, na.rm = TRUE),
            ss_nll = mean(ss, na.rm = TRUE), 
            sn_nll = mean(ssnn, na.rm = TRUE))
overall_comparisons
write_csv(overall_comparisons, 'out/nll-comps.csv')
