
# Fit movement models with momentuHMM -------------------------------------
library(tidyverse)
library(momentuHMM)
train_trajectories <- list.files("out/trajectories/train/", 
                                 pattern = "coords.csv", recursive = TRUE, 
                                 full.names = TRUE) %>%
  sort



get_gamma_params <- function(fit) {
  # compute gamma shape and rate from mean and sd
  mean_sd_matrix <- fit$mle$step
  pars <- apply(mean_sd_matrix, 2, function(x) {
    c((x["mean"] / x["sd"])^2, x["mean"] / x["sd"]^2)
  })
  rownames(pars) <- c("gamma_shape", "gamma_rate")
  as_tibble(t(pars)) # transpose to match order expected in forward algorithm function
}

get_vm_params <- function(fit) {
  pars <- t(fit$mle$angle)
  colnames(pars) <- c("vm_mean", "vm_concentration")
  pars <- as_tibble(pars)
  pars
}

get_fixef <- function(fit) {
  pars <- t(fit$mle$beta)
  rown <- rownames(pars)
  pars <- as_tibble(pars)
  pars$transition <- rown
  pars %>%
    rename(transition_intercept = `(Intercept)`)
}

get_start_probs <- function(fit) {
  p <- as_tibble(fit$mle$delta)
  names(p) <- gsub(" ", "", names(p))
  p
}

save_output <- function(fit, name) {
  dir.create("out/params", showWarnings = FALSE)
  get_gamma_params(fit) %>%
    bind_cols(get_vm_params(fit)) %>%
    bind_cols(get_fixef(fit)) %>%
    mutate(state = 1:n()) %>%
    write_csv(path = paste0("out/params/", name, "_", sample_size, ".csv"))
}

sample_sizes <- 2^c(4:10)

for (sample_size in sample_sizes) {
  print(paste("Fitting baseline movement models with sample size", sample_size))
  data <- train_trajectories[1:sample_size] %>%
    lapply(read_csv, col_types = cols()) %>%
    bind_rows(.id = "ID") %>%
    mutate_at(vars(starts_with("rgb")), ~ . / 255)
  
  d <- prepData(as.data.frame(data), 
                type = "UTM", 
                coordNames = c("x", "y"), 
                covNames = "z")
  
  # initial step distribution natural scale parameters
  stepPar0 <- c(1, 5, 0.5, 3) # (mu_1,mu_2,sd_1,sd_2)
  # initial angle distribution natural scale parameters
  anglePar0 <- c(0, 0, 1, 8) # (mean_1,mean_2,concentration_1,concentration_2)
  fit <- fitHMM(data = d, 
                nbStates = 2,
                dist = list(step = "gamma", angle = "vm"),
                Par0 = list(step = stepPar0, angle = anglePar0),
                formula = ~ z,
                formulaDelta = ~z,
                estAngleMean = list(angle=TRUE))
  
  save_output(fit, name = "bestcase")
  
  rgb_d <- prepData(as.data.frame(data), type = "UTM", coordNames = c("x", "y"), 
                    covNames = c("rgb_mosaic.1", "rgb_mosaic.2", "rgb_mosaic.3"))
  
  rgb_fit <- fitHMM(data = d, 
                    nbStates = 2,
                    dist = list(step = "gamma", angle = "vm"),
                    Par0 = list(step = stepPar0, angle = anglePar0),
                    formula = ~ rgb_mosaic.1 + rgb_mosaic.2 + rgb_mosaic.3,
                    formulaDelta = ~ rgb_mosaic.1 + rgb_mosaic.2 + rgb_mosaic.3,
                    estAngleMean = list(angle=TRUE))
  
  save_output(rgb_fit, name = "ptextract")
}
