library(tidyverse)
library(rstan)
rstan_options(auto_write = TRUE)
library(pbapply)
library(parallel)
library(vroom)

bbs <- vroom('data/cleaned/bbs.csv') %>% 
  left_join(read_csv('data/cleaned/clean_routes.csv')) %>%
  mutate(f_l1 = factor(L1_KEY)) %>%  
  split(.$aou)

m_init <- stan_model('stan/dynamic-occupancy.stan')

fit_species_model <- function(orig_df, model, overwrite = FALSE) {
  out_path <- unique(orig_df$english) %>%
    tolower %>%
    gsub(' ', '_', x = .) %>%
    paste0(., '_ss.csv') %>%
    file.path('out', .)
  
  if (file.exists(out_path) & !overwrite) {
    return(NULL)
  }
  
  df <- orig_df %>%
    filter(group != 'test')
  
  train_d <- orig_df %>%
    filter(group == 'train') %>%
    arrange(route_id)
  
  Y <- train_d %>%
    select(starts_with('1'), starts_with('2')) %>%
    as.matrix
  
  make_X_p <- function(df) {
    df %>%
      select(starts_with('Start'), starts_with('End'), starts_with('duration'), 
             'route_id') %>%
      gather(var, value, -route_id) %>%
      mutate(value = ifelse(is.na(value), 0, value)) %>%   
      separate(var, c('var', 'year'), sep = '_') %>%
      spread(var, value) %>%
      arrange(route_id) %>%
      split(.$year) %>%
      lapply(FUN = function(x) {
        model.matrix(~ duration + EndSky + EndTemp + EndWind + 
                       StartSky + StartTemp + StartWind, data = x)
      })
  }
  
  X_p <- make_X_p(df = train_d)
  X_p_pred <- make_X_p(df = filter(df, group != 'test'))
  
  formula <- ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + 
    c_lat + c_lon + c_elevation + c_road_den + c_dist_shore
  X <- model.matrix(formula, data = train_d)
  X_pred <- model.matrix(formula, data = filter(df, group != 'test'))
  
  any_surveys <- as.numeric(!is.na(Y)) %>%
    matrix(nrow = nrow(Y), ncol = ncol(Y))
  
  Y_full <- Y
  Y_full[!any_surveys] <- -1
  
  stopifnot(!any(is.na(Y_full)))
  
  stan_d <- list(
    nsite = nrow(Y), 
    nyear = ncol(Y), 
    nrep = 50, 
    Y = Y_full, 
    any_surveys = any_surveys, 
    n_l1 = length(unique(orig_df$L1_KEY)), 
    l1 = as.numeric(train_d$f_l1), 
    n_pred = nrow(df),
    l1_pred = as.numeric(df$f_l1), 
    m = ncol(X), 
    X = X, 
    X_pred = X_pred, 
    m_p = ncol(X_p[[1]]), 
    X_p = X_p,
    X_p_pred = X_p_pred
  )
  
  n_tries <- 10
  m_fit <- list(return_code = -1)
  while(n_tries > 0 & m_fit$return_code != 0) {
      m_fit <- optimizing(model, data = stan_d, verbose = TRUE, 
                          as_vector = FALSE, iter=1e5, 
                          init = list(
                            # intercepts and slopes initialize differently
                            alpha = c(-4, -4, -4, -4),  
                            beta = matrix(0, nrow = stan_d$m, ncol = 4),
                            z_l1 = matrix(0, nrow = stan_d$n_l1, ncol = 4), 
                            sigma_l1 = rep(.01, 4)
                            )
                          )

    n_tries = n_tries - 1
  }
  if (m_fit$return_code != 0) {
    warning(paste0("Model did not converge for", 
                   unique(df$english)))
  }
  
  years <- colnames(Y)

  colnames(m_fit$par$p_pred) <- paste0('p_', years)
  p_df <- as_tibble(m_fit$par$p_pred, .name_repair = 'unique')
  gamma_df <- tibble(gamma = m_fit$par$gamma_pred)
  phi_df <- tibble(phi = m_fit$par$phi_pred)
  
  tibble(psi1 = m_fit$par$psi1_pred) %>%
    bind_cols(p_df) %>%
    bind_cols(gamma_df) %>%
    bind_cols(phi_df) %>%
    bind_cols(df) %>%
    mutate(return_code = m_fit$return_code) %>%
    write_csv(out_path)
}


pboptions(use_lb=TRUE)
cl <- makeCluster(parallel::detectCores() - 1)
clusterEvalQ(cl, library(tidyverse))
clusterEvalQ(cl, library(rstan))
out <- pblapply(bbs, fit_species_model, cl = cl, model = m_init)
stopCluster(cl)
