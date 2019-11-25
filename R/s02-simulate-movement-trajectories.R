library(circular)
library(data.table)
library(raster)
library(tidyverse)
library(sf)
library(pbapply)
library(patchwork)


# Load the CHM and RGB data -----------------------------------------------

chm <- raster("out/chm_mosaic.tif")
mean(values(chm) > 30)
chm[chm > 30] <- 30
chm <- chm / cellStats(chm, max)

rgb_mosaic <- stack("out/rgb_mosaic.tif")

# plot a subset of the chm next to the rgb mosaic
crop_ext <- raster::extent(254750, 255000, 4107750, 4108000)
rgb_chip <- stack("data/DP3.30010.001/2018/FullSite/D17/2018_SJER_3/L3/Camera/Mosaic/2018_SJER_3_254000_4107000_image.tif") %>%
  crop(crop_ext)
chm_chip <- raster("data/DP3.30015.001/2018/FullSite/D17/2018_SJER_3/L3/DiscreteLidar/CanopyHeightModelGtif/NEON_D17_SJER_DP3_254000_4107000_CHM.tif") %>%
  crop(crop_ext)
png('fig/chm-rgb.png', width = 1000, height = 600)
par(mfrow = c(1, 2))
plot(chm_chip, col = viridis::viridis(100), axes = FALSE, 
     box = FALSE, legend = FALSE)
plotRGB(rgb_chip, margins = TRUE)
par(mfrow = c(1, 1))
dev.off()


# Simulate transition probabilities -----------------------------------------
gamma_12 <- function(h) {
  # transition from in transit to foraging
  # more likely if chm is high
  plogis(-6 + 40 * h)
}

gamma_21 <- function(h) {
  # transition from foraging to in transit
  # more likely if you're outside the chm
  plogis(6 - 40 * h)
}

transition_df <- tibble(chm = seq(0, 1, by = .01), 
                        gamma_12 = gamma_12(chm), 
                        gamma_21 = gamma_21(chm)) %>%
  pivot_longer(starts_with("gamma"), "par") %>%
  mutate(Parameter = ifelse(par == "gamma_12", 
                            "Transition to \"foraging\"", 
                            "Transition to \"in transit\""))

trans_plot <- transition_df %>%
  ggplot(aes(chm, value, color = Parameter)) + 
  geom_line() + 
  xlab("Canopy height") + 
  ylab("Probability") +
  geom_text(data = filter(transition_df, chm == 0.5), 
            aes(label = Parameter, y = value + .1)) + 
  theme_minimal() + 
  theme(legend.position = 'none', 
        panel.grid.minor = element_blank()) + 
  scale_color_manual(values = c("black", "red")) + 
  scale_y_continuous(breaks = seq(0, 1, by = .25)) + 
  ggtitle("A")
trans_plot


# Visualize step size distributions by behavioral state --------------------
sim_step_size <- function(n, state) {
  if (state == 1) {
    x <- rgamma(n, 10, 1)
  } else if (state == 2) {
    x <- rgamma(n, 10, 5)
  }
  return(x)
}

gamma_density_df <- tibble(x = seq(0, 25, by = .1), 
                           d1 = dgamma(x, 10, 1), 
                           d2 = dgamma(x, 10, 5)) %>%
  pivot_longer(starts_with("d")) %>%
  mutate(state = ifelse(name == "d1", "In transit", "Foraging"))

step_size_plot <- gamma_density_df %>%
  ggplot(aes(x, value, color = state)) + 
  geom_line() + 
  theme_minimal() + 
  xlab("Step size (m)") + 
  ylab("Density") + 
  theme(panel.grid.minor = element_blank()) + 
  scale_color_manual(values = c("red", "black")) + 
  annotate("text", x = 7, y = .6, label = "Foraging", color = "red") + 
  annotate("text", x = 15, y = .15, label = "In transit", color = "black") + 
  theme(legend.position = "none") + 
  ggtitle("B")
step_size_plot



# Visualize Von Mises turn angle density by behavioral state --------------
sim_radii <- function(n, state) {
  if (state == 1) {
    r <- rvonmises(n, mu = 0, kappa = 20)
  } else if (state == 2) {
    r <- rvonmises(n, mu = 0, kappa = .1)
  }
  return(r)
}

get_vm_df <- function() {
  ff1 <- function(x) dvonmises(x, mu=circular(0), kappa=20)
  curve1_data <- curve.circular(ff1, join=TRUE, n = 200)
  ff2 <- function(x) dvonmises(x, mu=circular(0), kappa=.1)
  curve2_data <- curve.circular(ff2, join=TRUE, n = 200)
  
  vm1_df <- tibble(x = curve1_data$x, 
                   y = curve1_data$y, 
                   state = "In transit")
  vm2_df <- tibble(x = curve2_data$x, 
                   y = curve2_data$y, 
                   state = "Foraging")
  
  vm_df <- vm1_df %>%
    full_join(vm2_df)
  vm_df
}

vm_df <- get_vm_df()
vm_plot <- vm_df %>%
  ggplot(aes(x, y, color = state)) + 
  geom_segment(aes(x = x, xend = xend, y = y, yend = yend), 
               data = tibble(x = c(-1, 0), 
                             xend = c(1, 0), 
                             y = c(0, -1), 
                             yend = c(0, 1)), 
               alpha = .1, inherit.aes = FALSE) +
  annotate("path",
           x = cos(seq(0,2*pi,length.out=100)),
           y = sin(seq(0,2*pi,length.out=100)), 
           size = .1) + 
  geom_path() + 
  geom_point(x = 0, y = 0, color = "black") + 
  theme_minimal() + 
  theme(panel.grid = element_blank(),
        axis.text = element_blank(),
        axis.title = element_blank(),
        legend.position = "none") +
  annotate("text", 
           x = c(1, 0, -1, 0) * .8, 
           y = c(0, 1, 0, -1) * .8, 
           label = c(0, 
                     expression(paste(pi / 2)), 
                     expression(pi), 
                     expression(paste(3*pi/2)))) + 
  coord_equal() + 
  scale_color_manual(values = c("red", "black")) +
  ggtitle("C")
vm_plot  


# put the visualizations together
dist_plot <- trans_plot / (step_size_plot + vm_plot) + 
  plot_layout(heights = c(.7, 1))
dist_plot
ggsave("fig/movement-distributions.pdf", width = 6, height = 5)


# Simulate movement over the CHM ------------------------------------------

stationary_probs <- function(z) {
  # get the stationary probabilities for a particular value of CHM (z)
  g12 <- gamma_12(z)
  g21 <- gamma_21(z)
  cbind(g21, g12) / (g12 + g21)
}

get_coords <- function(x, y) {
  st_as_sf(tibble(easting = x, northing = y), 
           coords = c("easting", "northing"), 
           crs = "+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0", 
           agr = "constant")
}



get_num_traj <- function() {
  trajectories <- lapply(file.path("out", "trajectories", 
                                   c("test", "train", "validation")), 
                         list.files, full.names = TRUE)
  
  # how many trajectories are there in the test, train, and validation set?
  nfiles <- lapply(trajectories, length)
  names(nfiles) <- c("test", "train", "validation")
  unlist(nfiles)
}



simulate_trajectory <- function(dummy_arg) {
  
  if (all(get_num_traj() == 1024)) {
    return(NULL)
  }
  
  sim_id <- Sys.time() %>%
    format("%Y-%m-%d_%H%M%S") %>%
    paste(sep = "_", sample(letters, size = 8, replace = TRUE) %>% 
            paste(collapse=''))
  n_timesteps <- 50
  ex <- extent(chm)
  x_init <- runif(1, ex@xmin, ex@xmax)
  y_init <- runif(1, ex@ymin, ex@ymax)
  
  rads1 <- sim_radii(n_timesteps, state = 1)
  rads2 <- sim_radii(n_timesteps, state = 2)
  
  coords <- get_coords(x_init, y_init)
  
  z0 <- raster::extract(chm, coords)
  if (is.na(z0)) return(NULL)
  
  state <- rep(NA, n_timesteps + 1)
  state[1] <- 1 + rbinom(1, 1, stationary_probs(z0)[2])
  
  lengths1 <- sim_step_size(n_timesteps, state = 1)
  lengths2 <- sim_step_size(n_timesteps, state = 2)
  
  x <- rep(NA, n_timesteps + 1)
  y <- rep(NA, n_timesteps + 1)
  z <- rep(NA, n_timesteps + 1)
  turn_angle <- rep(NA, n_timesteps + 1)
  step_size <- rep(NA, n_timesteps + 1)
  
  x[1] <- x_init
  y[1] <- y_init
  z[1] <- z0
  
  # iterate over step sizes and turning angles to create path
  cumrads <- 0
  for (t in 1:n_timesteps) {
    if (state[t] == 1) {
      # state 1: in transit
      turn_angle[t] <- rads1[t]
      step_size[t] <- lengths1[t]
    } else {
      # state 2: foraging
      turn_angle[t] <- rads2[t]
      step_size[t] <- lengths2[t]
    }
    if (t == 1) {
      # for the initial timestep, choose a random starting direction
      turn_angle[t] <- rvonmises(1, 0, 0)
    }
    cumrads <- cumrads + turn_angle[t]
    
    x[t + 1] <- x[t] + step_size[t] * cos(cumrads)
    y[t + 1] <- y[t] + step_size[t] * sin(cumrads)
    xy <- get_coords(x[t + 1], y[t + 1])
    z[t + 1] <- raster::extract(chm, xy)
    
    if (is.na(x[t + 1]) | is.na(z[t + 1])) {
      return(NULL)
    }
    assertthat::assert_that(!is.na(x[t + 1]))
    assertthat::assert_that(!is.na(z[t + 1]))
    
    # deal with state transitions
    if (state[t] == 1) { # in transit
      change <- rbinom(1, 1, gamma_12(z[t + 1]))
      if (change == 1) {
        state[t+1] <- 2
      } else {
        state[t+1] <- state[t]
      }
    } else { # you're foraging
      change <- rbinom(1, 1, gamma_21(z[t + 1]))
      if (change == 1) {
        state[t+1] <- 1
      } else {
        state[t+1] <- state[t]
      }
    }
    if (is.na(change)) return(NULL)
    assertthat::assert_that(!is.na(change))
  }
  
  df <- tibble(x = x, y = y, z = z,
               turn_angle = turn_angle, step_size = step_size) %>%
    mutate(t = 1:n(), 
           state = state,
           state = ifelse(state == 1, "In transit", "Foraging")) %>%
    filter(t != n()) # remove terminal timestep, which has no movement data
  
  df_sf <- get_coords(df$x, df$y) %>%
    bind_cols(df)
  
  buffered_pts <- df_sf %>%
    st_buffer(dist = 6.45) %>% # buffer distance determines chip size
    mutate(t = 1:n())
  
  rgb_crop <- crop(rgb_mosaic, buffered_pts)
  
  chips <- buffered_pts  %>%
    split(.$t) %>%
    lapply(function(x) {
      r <- raster::crop(rgb_crop, raster::extent(x), snap = "in")
      r
    })
  
  chip_bboxes <- buffered_pts  %>%
    split(.$t) %>%
    lapply(function(x) {
      st_bbox(x) %>% st_as_sfc %>% st_sf
    })
  chip_bboxes <- sf::st_as_sf(data.table::rbindlist(chip_bboxes))
  
  rgb_extractions <- raster::extract(rgb_crop, df_sf)  %>%
    as_tibble %>%
    mutate(t=1:n())
  
  # verify that all chips have the same dimension (would be violated at edge)
  ncells <- lapply(chips, raster::ncell) %>%
    unlist
  if (length(unique(ncells)) != 1) {
    return(NULL)
  }
  assertthat::assert_that(length(unique(ncells)) == 1)
  
  n_na <- lapply(chips, function(x) {
    cellStats(is.na(x), sum)
  }) %>%
    unlist %>%
    sum
  assertthat::assert_that(n_na == 0)
  
  # assign the trajectory to the training, validation, or test set
  # these values correspond to splitting the northing values into thirds
  total_extent <- extent(rgb_mosaic)
  crop_extent <- extent(rgb_crop)
  
  y_breaks <- seq(total_extent@ymin, total_extent@ymax, length.out = 4)
  group <- case_when(
    crop_extent@ymax < y_breaks[2] ~ "test",
    crop_extent@ymin > y_breaks[3] ~ "train",
    TRUE ~ "validation"
  )
  
  # Write output to a directory to read later
  out_dir <- file.path("out", "trajectories", group, sim_id)
  
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  chip_paths <- rep(NA, length(chips))
  for (t in seq_along(chips)) {
    chip_paths[t] <- file.path(out_dir, 
                         paste0("chip_", 
                                sprintf("%03d", t),
                                ".tif"))
    # write a regular tiff
    writeRaster(chips[[t]], 
                filename = chip_paths[t],
                options = c("PROFILE=BASELINE"), # makes it not a geotiff
                datatype = 'INT1U')
    file.rename(chip_paths[t], paste0(chip_paths[t], 'f'))
  }

  probs <- stationary_probs(z)
  
  df_sf <- left_join(df_sf, rgb_extractions) %>%
    mutate(stationary_p1 = probs[1:n(), 1], 
           stationary_p2 = probs[1:n(), 2])
  
  df_path <- file.path(out_dir, "coords.csv")
  df_sf %>%
    as_tibble %>%
    dplyr::select(-geometry) %>%
    write_csv(df_path)
  
  sf_path <- file.path(out_dir, "coords.gpkg")
  df_sf %>%
    st_write(sf_path, quiet = TRUE)
  
  chip_path <- file.path(out_dir, "chip_bboxes.gpkg")
  chip_bboxes %>%
    st_write(chip_path, quiet = TRUE)
  
  rgb_path <- file.path(out_dir, "rgb_crop.tif")
  rgb_crop %>%
    writeRaster(rgb_path)
  
  list(df_sf = sf_path,
       rgb_crop = rgb_path, 
       out_files = list.files(out_dir, full.names = TRUE))
}

cl <- parallel::makeCluster(parallel::detectCores())
parallel::clusterEvalQ(cl, {
  library(circular)
  library(sf)
  library(raster)
  library(tidyverse)
})
parallel::clusterExport(cl, 
                        c("chm", "rgb_mosaic", "gamma_12", "gamma_21", 
                          "sim_step_size", "sim_radii", "stationary_probs", 
                          "get_coords", "get_num_traj"))
out <- pblapply(1:4000, simulate_trajectory, cl = cl)
parallel::stopCluster(cl)



# Sanity checks for simulated trajectories --------------------------------


# 1. if any more than 1024 trajectories per dir, delete the extras
lapply(file.path("out", "trajectories", c("test", "train", "validation")), 
       function(x) {
         f <- list.files(x, include.dirs = TRUE, full.names = TRUE)
         if (length(f) > 1024) {
           to_delete <- f[1025:length(f)]
           unlink(to_delete, recursive = TRUE, force = TRUE)
         }
       })



# 2. all written trajectories have 50 image chips
trajectories <- lapply(file.path("out", "trajectories", 
                                     c("test", "train", "validation")), 
                           list.files, full.names = TRUE)

lapply(unlist(trajectories), function(x) {
  chip_files <- list.files(path = x, pattern = "*.tiff")
  if (length(chip_files) != 50) {
    unlink(x, recursive = TRUE, force = TRUE)
  }
})

stopifnot(all(get_num_traj() == 1024))


# Generate an example plot for a movement trajectory ----------------------

trajdir <- sample(trajectories[[1]], size = 1)

sim_df <- file.path(trajdir, "coords.gpkg") %>%
  st_read


p1 <- sim_df %>%
  ggplot(aes(x, y)) + 
  geom_raster(data = chm %>%
                crop(sim_df) %>%
                as.data.frame(xy = TRUE) %>%
                as_tibble, 
              aes(fill = chm_mosaic), 
              alpha = .9) + 
  geom_segment(aes(x = x, y = y, xend = lead(x), yend = lead(y), 
                   color = state),
               arrow = arrow(length = unit(0.03, "npc"))) + 
  scale_fill_viridis_c("CHM") + 
  theme_minimal() + 
  scale_color_manual(values = c("red", "black"), "State") + 
  coord_equal() + 
  theme(axis.text = element_blank(), panel.grid = element_blank(), 
        legend.position = "left") + 
  xlab("") + 
  ylab("") + 
  ggtitle("A")
p1

chips <- file.path(trajdir, "chip_bboxes.gpkg") %>%
  st_read %>%
  mutate(id = 1:n()) %>%
  split(.$id) %>%
  pblapply(function(x) {
    stack(file.path(trajdir, "rgb_crop.tif")) %>%
      crop(x) 
  })

chip_mosaic <- chips
names(chip_mosaic)[1:2] <- c("x", "y")
chip_mosaic$fun <- mean
chip_mosaic$na.rm <- TRUE
final_chip_mosaic <- do.call(mosaic, chip_mosaic)

p2 <- ggplot() + 
  ggspatial::layer_spatial(data = final_chip_mosaic) + 
  theme_minimal() + 
  geom_sf(data = summarize(sim_df, do_union=FALSE) %>%
            st_cast('LINESTRING'), size = .1, color = "white") + 
  geom_sf(data = sim_df, color = "white", size = .4) + 
  theme(axis.text = element_blank(), 
        panel.grid = element_blank(), 
        legend.position = 'none') + 
  ggtitle("B")
p2

p1 + p2



# Visualize all chips in the training data
bbox_files <- list.files('out/trajectories', pattern = "chip_bboxes", 
                         recursive = TRUE, full.names = TRUE) %>%
  sort
bboxes <- bbox_files %>%
  lapply(st_read)

bbox_df <-  sf::st_as_sf(data.table::rbindlist(bboxes)) %>%
  mutate(src = rep(bbox_files, each = nrow(bboxes[[1]])), 
         group = case_when(
           grepl("train", src) ~ "train", 
           grepl("test", src) ~ "test", 
           grepl("valid", src) ~ "validation", 
           TRUE ~ "other"
         )) %>%
  group_by(src, group) %>%
  summarize()

traj_plot <- bbox_df %>%
  ungroup %>%
  mutate(group = factor(tools::toTitleCase(group), 
                        levels = c("Train", "Validation", "Test"))) %>%
  ggplot(aes(color = group)) + 
  geom_sf(size = .3, fill = NA) + 
  scale_color_manual("Partition", values = c("black", "dodgerblue", "red")) + 
  theme_minimal()
traj_plot
ggsave("fig/traj-plot.png", traj_plot, width = 8, height = 5)
