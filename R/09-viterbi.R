
# Compute most likely occupancy sequences ---------------------------------

library(vroom)
library(tidyverse)
library(pbapply)
library(parallel)
library(patchwork)
library(plotly)
library(ggrepel)
library(sf)
library(rmapshaper)

viterbi_alg <- function(df) {
  # Implement Viterbi algorithm to find most likely states ------------------
  # mostly follows the algorithm here: 
  # https://en.wikipedia.org/wiki/Viterbi_algorithm
  stopifnot(length(unique(df$route_id)) == 1)
  n_t <- nrow(df)
  df <- df %>%
    arrange(year) %>%
    mutate(logprob_y_present = ifelse(is.na(y), 0, 
                                      dbinom(y, 50, p, log = TRUE)), 
           logprob_y_absent = case_when(
             is.na(y) ~ 0, 
             y > 0 ~ -Inf,
             y == 0 ~ 0 # if absent, then you definitely observe a zero
           ))
  
  max_prob <- matrix(nrow = 2, ncol = n_t)
  max_state <- matrix(nrow = 2, ncol = n_t)
  
  # initial time step
  max_prob[1, 1] <- log(1 - df$psi[1]) + df$logprob_y_absent[1]
  max_prob[2, 1] <- log(df$psi[1]) + df$logprob_y_present[1]
  max_state[, 1] <- 0
  
  for (t in 2:n_t) {
    logprob_transitions <- log(matrix(c(1 - df$gamma[t - 1], 
                                        1 - df$phi[t - 1], 
                                        df$gamma[t - 1], 
                                        df$phi[t - 1]), 
                                      nrow = 2))
    for (s in 1:2) {
      if (s == 1) {
        # true state absent
        max_prob[s, t] <- max(max_prob[, t-1] + 
                                logprob_transitions[, s] + 
                                df$logprob_y_absent[t])
      } else {
        # true state present, s = 2
        max_prob[s, t] <- max(max_prob[, t-1] + 
                                logprob_transitions[, s] + 
                                df$logprob_y_present[t])
      }
      max_state[s, t] <- which.max(max_prob[, t-1] + 
                                     logprob_transitions[, s])
    }
  }
  
  z_T <- which.max(max_prob[, n_t])
  x <- rep(NA, n_t)
  x[n_t] <- z_T
  for (t in n_t:2) {
    x[t-1] <- max_state[x[t], t]
  }
  z <- x - 1

  df %>%
    mutate(z_mle = z) %>%
    select(route_id, sp.bbs, z_mle, year, psi, gamma, phi, p, y) %>%
    mutate(colonization_event = lag(z_mle == 0) & z_mle > 0, 
           extinction_event = lag(z_mle > 0) & z_mle == 0)
}

cl <- makeCluster(parallel::detectCores() / 2)
clusterEvalQ(cl, library(tidyverse))
clusterExport(cl, 'viterbi_alg')
z_mles <- list.files(path = 'out/q_dfs', full.names = TRUE) %>%
  pblapply(FUN = function(path) {
    read_csv(path) %>%
      split(.$route_id) %>%
      lapply(FUN = viterbi_alg) %>%
      bind_rows
  }, cl = cl) %>%
  bind_rows
stopCluster(cl)

write_csv(z_mles, path = 'out/z_mles.csv')



# Compute finite sample estimates -----------------------------------------

get_fs_estimates <- function(df) {
  # get finite sample estimates for one species
  stopifnot(length(unique(df$sp.bbs)) == 1)
  df %>%
    arrange(route_id, sp.bbs, year) %>%
    group_by(route_id) %>%
    mutate(z_mle_tplus1 = lead(z_mle), 
           z_mle_tminus1 = lag(z_mle), 
           mask_tp1 = is.na(z_mle_tplus1), 
           mask_tm1 = is.na(z_mle_tminus1)) %>%
    group_by(sp.bbs, year) %>%
    summarize(fs_psi = mean(z_mle),
              fs_growth_rate = sum(z_mle_tplus1[!mask_tp1]) / sum(z_mle[!mask_tp1]), 
              fs_turnover = sum((1 - z_mle_tminus1[!mask_tm1]) * z_mle[!mask_tm1]) / 
                            sum(z_mle[!mask_tm1])) %>%
    ungroup
}

  
cl <- makeCluster(parallel::detectCores())
pboptions(use_lb = TRUE)
clusterEvalQ(cl, library(tidyverse))
z_fs <- z_mles %>%
  split(.$sp.bbs) %>%
  pblapply(FUN = get_fs_estimates, cl = cl) %>%
  bind_rows
stopCluster(cl)
write_csv(z_fs, path = 'out/z_finite_sample.csv')
