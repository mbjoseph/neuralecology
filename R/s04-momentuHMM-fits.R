
# Fit movement models with momentuHMM -------------------------------------
library(tidyverse)
library(momentuHMM)
train_trajectories <- list.files("out/trajectories/train/", 
                                 pattern = "coords.csv", recursive = TRUE, 
                                 full.names = TRUE)

sample_size <- 128
data <- train_trajectories[1:sample_size] %>%
  lapply(read_csv) %>%
  bind_rows(.id = "ID")

d <- prepData(as.data.frame(data), type = "UTM", coordNames = c("x", "y"), 
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
              estAngleMean = list(angle=TRUE))

rgb_d <- prepData(as.data.frame(data), type = "UTM", coordNames = c("x", "y"), 
                  covNames = c("rgb_mosaic.1", "rgb_mosaic.2", "rgb_mosaic.3"))

rgb_fit <- fitHMM(data = d, 
                  nbStates = 2,
                  dist = list(step = "gamma", angle = "vm"),
                  Par0 = list(step = stepPar0, angle = anglePar0),
                  formula = ~ rgb_mosaic.1 + rgb_mosaic.2 + rgb_mosaic.3,
                  estAngleMean = list(angle=TRUE))
