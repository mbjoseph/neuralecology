library(tidyverse)
library(raster)
library(ggspatial)
library(sf)
library(patchwork)

chm <- raster("out/scaled_chm_mosaic.tif")


test_preds <- read_csv("notebooks/test-set-checks.csv") %>%
  mutate(gamma_12 = plogis(6 - 40 * chm), 
         gamma_21 = plogis(-6 + 40 * chm),
         g12_diff = pred_gamma_12 - gamma_12, 
         g21_diff = pred_gamma_21 - gamma_21)

test_preds %>%
  ggplot(aes(gamma_12, pred_gamma_12)) + 
  geom_jitter(alpha = .1, width = .03, height = 0, size = .5) + 
  geom_abline(linetype = "dashed") + 
  xlab(expression(paste("True ", gamma["1,2"]))) + 
  ylab(expression(paste("Estimated ", gamma["1,2"])))

test_preds %>%
  ggplot(aes(gamma_21, pred_gamma_21)) + 
  geom_jitter(alpha = .1, width = .03, height = 0, size = .5) + 
  geom_abline(linetype = "dashed") + 
  xlab(expression(paste("True ", gamma["2,1"]))) + 
  ylab(expression(paste("Estimated ", gamma["2,1"])))



# Plot some rows of interest -----------------------------------------------
plot_df_row <- function(df) {
  stopifnot(nrow(df) == 1)
  
  dir <- file.path("out", "trajectories", "test", df$directory)
  chip <- dir %>%
    list.files(pattern = sprintf("%03d", df$t), full.names = TRUE) %>%
    grep("tiff$", ., value = TRUE) %>%
    raster::stack()
  bbox <- st_read(file.path(dir, "chip_bboxes.gpkg"))[df$t, ]
  chm_chip <- raster::crop(chm, bbox)
  rgb_plot <- ggplot() + 
    ggspatial::layer_spatial(data = chip) + 
    theme_void()
  
  # chm_plot <- ggplot() + 
  #   geom_raster(data = as_tibble(as.data.frame(chm_chip, xy = TRUE)), 
  #               aes(x, y, fill=scaled_chm_mosaic)) + 
  #   theme_void() +
  #   scale_fill_viridis_c(limits = c(0, 1), 
  #                        oob = scales::squish) +
  #   theme(legend.position = "none") +
  #   coord_equal()
  rgb_plot #/ chm_plot
}


# good - maximum predicted probability of transitioning to in transit where chm=0
test_preds %>%
  filter(chm == 0, pred_gamma_12 == max(pred_gamma_12)) %>%
  plot_df_row

# bad - minimum predicted probability of transitioning to in transit where chm=0
test_preds %>%
  filter(chm == 0) %>%
  filter(pred_gamma_12 == min(pred_gamma_12)) %>%
  plot_df_row

# good - maximum probability of transitioning to foraging where chm>0
test_preds %>%
  filter(chm != 0, pred_gamma_21 == max(pred_gamma_21)) %>%
  plot_df_row

# bad - minimum probability of transitioning to foraging where chm>0
test_preds %>%
  filter(chm != 0) %>%
  filter(pred_gamma_21 == min(pred_gamma_21)) %>%
  plot_df_row


plots <- test_preds %>%
  top_n(16, pred_gamma_12) %>%
  arrange(pred_gamma_12) %>%
  mutate(id = 1:n()) %>%
  split(.$id) %>%
  lapply(plot_df_row)
p1 <- wrap_plots(plots) + plot_annotation(title="(a)")


plots_21 <- test_preds %>%
  top_n(16, pred_gamma_21) %>%
  arrange(pred_gamma_21) %>%
  mutate(id = 1:n()) %>%
  split(.$id) %>%
  lapply(plot_df_row)
p2 <- wrap_plots(plots_21) + plot_annotation(title='(b)')

p1/p2

