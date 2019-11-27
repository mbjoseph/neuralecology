library(tidyverse)
library(pbapply)
library(parallel)
library(assertthat)
library(sf)
library(pROC)
library(patchwork)
library(vroom)
library(yardstick)
library(rmapshaper)
library(here)
source('R/utils.R')

theme_set(theme_minimal() + 
            theme(panel.grid.minor = element_blank()))

species <- list.files('out', '_finalnet.csv', full.names = TRUE) %>%
  gsub('out/', '', .) %>%
  gsub('\\_finalnet.csv', '', .)

dir.create(here::here('out', 'q_dfs'), showWarnings = FALSE, recursive = TRUE)

load_final_fit <- function(sp) {
  routes <- read_csv('data/cleaned/clean_routes.csv')
    
  nnet_fit <- here::here('out', paste0(sp, '_finalnet.csv')) %>%
    read_csv %>%
    mutate(method = 'nn') %>%
    left_join(select(routes, route_id, group))

  y_obs <- nnet_fit %>%
    select(sp.bbs, route_id, group, as.character(1997:2017)) %>%
    gather(var, value, -sp.bbs, -route_id, -group) %>%
    mutate(var = ifelse(!grepl("_", var), 
                        paste('y', var, sep = "_"), 
                        var)) %>%
    separate(var, into = c('var', 'year'), sep = '_') %>%
    rename(y = value) %>%
    select(-var) %>%
    mutate(sp.bbs = as.character(sp.bbs))

  calc_psi <- function(df) {
    for (i in 2:nrow(df)) {
      df$psi[i] <- df$psi[i - 1] * df$phi[i - 1] + 
        (1 - df$psi[i - 1]) * df$gamma[i - 1]
    }
    df
  }
    
  psi_df <- nnet_fit %>%
    select(sp.bbs, route_id, group, starts_with("p_"), starts_with("phi_"), 
           starts_with("gamma_"), starts_with("psi")) %>%
    gather(var, value, -sp.bbs, -route_id, -group) %>%
    mutate(var = ifelse(var == "psi0", "psi_1997", var)) %>%
    separate(var, into = c("var", "year")) %>%
    spread(var, value) %>%
    arrange(sp.bbs, route_id, year) %>%
    unite("sp_route", sp.bbs, route_id, sep = "__") %>%
    split(.$sp_route) %>%
    lapply(FUN = calc_psi) %>%
    bind_rows() %>%
    separate(sp_route, into = c('sp.bbs', 'route_id'), sep = "__")
    
  quant_df <- psi_df %>%
    mutate(marginal_pred = psi * qbinom(.5, size = 50, prob = p),
           conditional_pred = ifelse(psi > .5, 
                                     qbinom(.5, size = 50, prob = p), 
                                     0), 
           conditional_lo = ifelse(psi > .5, 
                                   qbinom(.025, size = 50, prob = p), 
                                   0), 
           conditional_hi = ifelse(psi > .5, 
                                   qbinom(.975, size = 50, prob = p), 
                                   0), 
           phi = phi, 
           gamma = gamma) %>%
    left_join(y_obs)
  
  out_name <- here::here('out', 'q_dfs', paste0(quant_df$sp.bbs[1], '.csv'))
  write_csv(quant_df, out_name)
  
  quant_df
}

test <- load_final_fit(species[1])

# write out q_dfs for each species
dir.create(here::here('out', 'q_dfs'), showWarnings = FALSE)
pboptions(use_lb = TRUE)
cl <- makeCluster(parallel::detectCores())
clusterEvalQ(cl, library(tidyverse))
ll_dfs <- pblapply(species, load_final_fit, cl = cl)
stopCluster(cl)



# To evaluate predictive performance, remove years with no surveys
q_df <- ll_dfs %>%
  bind_rows %>%
  filter(!is.na(y), group == 'test')

# compute prob of zero and prob of nonzero
q_df <- q_df %>%
  mutate(pr_zero = (1 - psi) + psi * (1 - p)^50, 
         estimate = 1 - pr_zero, 
         truth = factor(ifelse(y > 0, 'detected', 'not detected')))

roc_df <- q_df %>%
  group_by(route_id) %>%
  roc_curve(truth, estimate) %>%
  ungroup

auc_df <- q_df %>%
  group_by(route_id) %>%
  roc_auc(truth, estimate)


roc_plot <- roc_df %>%
  left_join(auc_df) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, group = route_id, 
             color = .estimate)) +
  geom_path(alpha = .5, size = .1) +
  geom_abline(lty = 3) +
  theme_minimal() + 
  theme(panel.grid.minor = element_blank(), legend.position = 'none') + 
  xlab("1 - Specificity") + 
  ylab("Sensitivity") + 
  ggtitle('(a)') +
  scale_color_gradient(low = 'red', high = 'dodgerblue', 'AUC')

auc_plot <- auc_df %>%
  ggplot(aes(.estimate)) + 
  geom_histogram(bins=50) + 
  xlab('AUC') + 
  ylab("Frequency") + 
  ggtitle('(b)')

routes <- st_read('data/cleaned/routes.shp') %>%
  left_join(auc_df) %>%
  st_transform(3174)

ecoregions <- st_read('data/NA_CEC_Eco_Level3.shp') %>%
  st_transform(3174) %>%
  ms_simplify %>%
  st_crop(routes)

l2_regions <- ecoregions %>%
  group_by(NA_L2KEY) %>%
  summarize

auc_map <- routes %>%
  ggplot() +
  geom_sf(data = l2_regions, fill = 'white', size = .1, alpha = .7, 
          color ='grey') +
  geom_sf(alpha = .2, size = .1) + 
  geom_sf(aes(color = .estimate), 
          data = filter(routes, !is.na(.estimate)), size = .4) +
  scale_color_gradient(low = 'red', high = 'dodgerblue', 'AUC') + 
  ggtitle('(c)')
auc_map

p <- (roc_plot | auc_plot) / auc_map + plot_layout(heights = c(.5, 1))

ggsave(filename = 'fig/roc-test.jpg', plot = p, width = 5, height = 5)


write_csv(auc_df, 'out/auc_df.csv')


# Compute and save test set interval coverage stats -----------------------

coverage_df <- q_df %>%
  mutate(y_less_than_pred = y < conditional_lo,
         y_more_than_pred = y > conditional_hi,
         y_in_interval = conditional_lo <= y & y <= conditional_hi) %>%
  group_by(route_id) %>%
  summarize(coverage = mean(y_in_interval), 
            y_below = mean(y_less_than_pred), 
            y_above = mean(y_more_than_pred))
write_csv(coverage_df, 'out/coverage_df.csv')
