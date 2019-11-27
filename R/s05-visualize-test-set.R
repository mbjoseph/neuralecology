library(tidyverse)
library(raster)
library(ggspatial)
library(sf)
library(patchwork)
library(ggExtra)
library(cowplot)

chm <- raster("out/scaled_chm_mosaic.tif")

theme_set(theme_minimal() + 
            theme(panel.grid.minor = element_blank()))

test_preds <- read_csv("out/test-set-checks.csv") %>%
  mutate(gamma_12 = plogis(-6 + 40 * chm), 
         gamma_21 = plogis(6 - 40 * chm))

# final model switched state order (so we plot true gamma_12 vs. estimated gamma_21)
# (there are no constraints to guarantee that state 1 or 2 is "in transit" or "foraging")
p1 <- test_preds %>%
  ggplot(aes(gamma_12, pred_gamma_21)) + 
  geom_point(color = NA) +
  stat_density_2d(geom = "raster", aes(fill = stat(density)), contour = FALSE) +
  scale_fill_viridis_c(option = "A") +
  xlab(expression(paste("True ", gamma["1,2"]))) + 
  ylab(expression(paste("Estimated ", gamma["1,2"]))) + 
  theme(legend.position = "none") + 
  coord_equal() + 
  ggtitle("(a)")
p1m <- ggMarginal(p1, type = "histogram")

p2 <- test_preds %>%
  ggplot(aes(gamma_21, pred_gamma_12)) + 
  geom_point(color = NA) +
  stat_density_2d(geom = "raster", aes(fill = stat(density)), contour = FALSE) +
  scale_fill_viridis_c(option = "A") +
  xlab(expression(paste("True ", gamma["2,1"]))) + 
  ylab(expression(paste("Estimated ", gamma["2,1"]))) + 
  theme(legend.position = "none") + 
  coord_equal() + 
  ggtitle("(b)")
p2m <- ggMarginal(p2, type = "histogram")

transition_densities <- cowplot::plot_grid(p1m, p2m)
ggsave("fig/transition-densities.png", plot = transition_densities, width = 6, height = 3.5)

# Plot some rows of interest -----------------------------------------------
plot_df_row <- function(df) {
  stopifnot(nrow(df) == 1)
  
  dir <- file.path("out", "trajectories", "test", df$directory)
  chip <- dir %>%
    list.files(pattern = sprintf("%03d", df$t), full.names = TRUE) %>%
    grep("chip.*.tiff$", ., value = TRUE) %>%
    raster::stack()
  bbox <- st_read(file.path(dir, "chip_bboxes.gpkg"))[df$t, ]
  
  rgb_plot <- ggplot() + 
    ggspatial::layer_spatial(data = chip) + 
    theme_void()
  rgb_plot
}

plots <- test_preds %>%
  top_n(8, pred_gamma_12) %>%
  arrange(pred_gamma_12) %>%
  mutate(id = 1:n()) %>%
  split(.$id) %>%
  lapply(plot_df_row)
plots[[1]] <- plots[[1]] + ggtitle("(a) Highest Pr(transition to \"in transit\")")

plots_21 <- test_preds %>%
  top_n(8, pred_gamma_21) %>%
  arrange(pred_gamma_21) %>%
  mutate(id = 1:n()) %>%
  split(.$id) %>%
  lapply(plot_df_row)
plots_21[[1]] <- plots_21[[1]] + ggtitle("(b) Highest Pr(transition to \"foraging\")")

top_prob_plots <- wrap_plots(c(plots, plots_21), nrow = 2)
ggsave("fig/top-prob-chips.png", top_prob_plots, width = 8, height = 2)
