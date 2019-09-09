library(vroom)
library(tidyverse)
library(patchwork)
library(plotly)
library(sf)
library(rmapshaper)
library(ggrepel)
library(pbapply)
library(multidplyr)

# Work with MLEs for latent states ----------------------------------------
# First, load the output from the Viterbi algorithm, route, and species data
z_mles <- vroom('out/z_mles.csv')
stopifnot(nrow(z_mles) > 1)
z_fs <- vroom('out/z_finite_sample.csv')
bbs_routes <- read_csv('data/cleaned/clean_routes.csv')
bbs_species <- read_csv('data/cleaned/bbs_species.csv')

epsg <- 102008

routes_sf <- st_read('data/cleaned/routes.shp') %>%
  st_transform(epsg)

ecoregions <- st_read("data/NA_CEC_Eco_Level3.shp") %>%
  ms_simplify(keep = 0.01) %>%
  st_transform(epsg) %>%
  summarize %>%
  st_crop(routes_sf)


# Quantify range centroids by species and year ----------------------------
# Restrict focus to well-sampled routes to avoid bias caused by an increase
# in Canadian route sampling.
# Also focus on common species, to avoid too much noise in the estimated
# centroid locations

routes_surveyed_most_years <- z_mles %>%
  filter(!is.na(y)) %>%
  group_by(route_id) %>%
  summarize(n_year = length(unique(year))) %>%
  ungroup %>%
  mutate(max_years = max(n_year)) %>%
  filter(n_year >= max_years)


common_sp_present_all_years <- z_mles %>%
  filter(z_mle == 1, route_id %in% routes_surveyed_most_years$route_id) %>%
  group_by(sp.bbs) %>%
  summarize(n_year = length(unique(year)), 
            n_routes = length(unique(route_id))) %>%
  ungroup %>%
  mutate(max_years = max(n_year)) %>%
  filter(n_year == max_years, 
         n_routes >= 100) %>%
  left_join(distinct(bbs_species, sp.bbs, english))


wtc <- function(g, w){
  # compute weighted centroids
  # https://github.com/r-spatial/sf/issues/977
  if (!(is(g,"sf")) | !(w %in% colnames(g))){
    stop(paste("requires an sf object with at a column",w))
  }
  centers = st_coordinates(st_centroid(st_geometry(g)))
  # crsx = st_crs(g) how could i reuse the CRS of g? do i need that?
  out = st_point(c(weighted.mean(centers[,"X"], g[[w]]), 
                   weighted.mean(centers[,"Y"], g[[w]])))
  return(st_sf(st_geometry(out)))
}

centroid_pts <- z_mles %>%
  filter(z_mle == 1) %>%
  filter(sp.bbs %in% common_sp_present_all_years$sp.bbs, 
         route_id %in% routes_surveyed_most_years$route_id) %>%
  left_join(bbs_routes) %>%
  st_as_sf(coords = c("Longitude", "Latitude"), 
           crs = 4326, agr = "constant") %>%
  st_transform(epsg) %>%
  group_by(year, sp.bbs) %>%
  summarize %>%
  st_centroid



# Generate species specific trajectories
species_paths <- centroid_pts %>%
  group_by(sp.bbs) %>%
  summarize(distance_gap = st_distance(
    geometry[year == 1997], 
    geometry[year == 2018], 
    by_element = TRUE)) %>% 
  st_cast("LINESTRING") %>%
  left_join(bbs_species)


growth_df <- z_fs %>%
  left_join(as_tibble(species_paths) %>%
              left_join(bbs_species) %>%
              select(sp.bbs, english, distance_gap)) %>%
  filter(!is.na(distance_gap)) %>%
  group_by(english) %>%
  summarize(dist_m = mean(as.numeric(distance_gap)), 
            growth_rate = mean(fs_growth_rate, na.rm = TRUE), 
            turnover = mean(fs_turnover, na.rm = TRUE)) %>%
  mutate(growing = growth_rate > 1)


select_paths <- species_paths %>%
  left_join(growth_df) %>%
  top_n(4, growth_rate) %>%
  arrange(-distance_gap) %>%
  mutate(english = reorder(english, -distance_gap))

write_csv(select_paths, 'out/select_paths.csv')


species_pts <- z_mles %>%
  filter(z_mle  == 1) %>%
  filter(sp.bbs %in% select_paths$sp.bbs) %>%
  group_by(sp.bbs, route_id) %>%
  summarize(min_year = min(year)) %>%
  ungroup %>%
  left_join(bbs_routes) %>%
  st_as_sf(coords = c("Longitude", "Latitude"), 
           crs = 4326, agr = "constant") %>%
  st_transform(epsg) %>%
  left_join(bbs_species) %>%
  left_join(distinct(as.data.frame(select_paths), 
                     english, distance_gap)) %>%
  mutate(english = reorder(english, -distance_gap))

species_endpoints <- select_paths %>%
  st_line_sample(sample = 0) %>%
  st_cast("POINT") %>%
  st_sf %>%
  mutate(english = select_paths$english)

species_startpoints <- select_paths %>%
  st_line_sample(sample = 1) %>%
  st_cast("POINT") %>%
  st_sf %>%
  mutate(english = select_paths$english)

displacement_map <- select_paths %>%
  mutate(english = fct_reorder(english, -growth_rate)) %>%
  ggplot() +  
  geom_sf(data = ecoregions, 
          fill = 'white',  
          size =.1, alpha = .9) +
  geom_sf(data = species_pts, alpha = .07, size = .1) +
  geom_sf_text(data = species_endpoints, label = "2018", size = 3, fontface = "bold") + 
  geom_sf_text(data = species_startpoints, label = "1997", size = 3, fontface = "bold") + 
  facet_wrap(~english, nrow = 2) + 
  xlab("") + 
  ylab("") + 
  theme_minimal()
displacement_map



# Relate centroid displacement to pop. growth rate ------------------------

to_label <- filter(growth_df, english %in% select_paths$english)

growth_plot <- growth_df %>%
  ggplot(aes(growth_rate, dist_m / 1000, group = english)) + 
  geom_point(alpha = .5, size = .5) + 
  scale_y_log10() + 
  geom_point(data = to_label) + 
  geom_text_repel(aes(label = english), 
                  data = to_label, size = 2.5) + 
  xlab("Population growth rate") + 
  ylab("Centroid displacement (km)") + 
  theme_minimal() + 
  theme(panel.grid.minor = element_blank(), 
        legend.position = 'none')
growth_plot


cd_plot <- (growth_plot + ggtitle("(a)") + coord_flip()) / 
  (displacement_map + ggtitle("(b)")) + 
  plot_layout(height = c(.5, 1)) 

dpi <- 150

cd_plot %>%
  {
    ggsave(filename = 'fig/centroid-displacement.jpg', plot = ., 
           width = 6, height = 7, dpi = dpi)
    ggsave(filename = 'fig/centroid-displacement.pdf', plot = ., 
           width = 6, height = 7)
    
  }




# Persistence probability as a function of distance from centroid -----------
route_pts <- z_mles %>%
  filter(z_mle == 1,
         sp.bbs %in% common_sp_present_all_years$sp.bbs, 
         route_id %in% routes_surveyed_most_years$route_id) %>%
  left_join(bbs_routes) %>%
  st_as_sf(coords = c("Longitude", "Latitude"), 
           crs = 4326, agr = "constant") %>%
  st_transform(epsg)


overall_bbs_centroid_dist <- routes_sf %>%
  ungroup %>%
  filter(route_id %in% routes_surveyed_most_years$route_id) %>%
  summarize() %>%
  st_centroid() %>%
  st_distance(centroid_pts) %>%
  c %>%
  as.numeric()

centroid_pts$overall_bbs_centroid_dist <- overall_bbs_centroid_dist

bbs_range_centroids <- centroid_pts %>%
  st_coordinates %>%
  as_tibble %>%
  mutate(year = centroid_pts$year, 
         sp.bbs = centroid_pts$sp.bbs)


cent_distances <- route_pts %>%
  left_join(bbs_range_centroids) %>%
  rowwise() %>%
  mutate(center_point = st_geometry(st_point(c(X, Y)))) %>%
  ungroup 

st_crs(cent_distances$center_point) <- epsg

dist_decay_df <- cent_distances %>%
  select(route_id, sp.bbs, z_mle, year, geometry, center_point, phi) %>%
  partition(sp.bbs) %>%
  mutate(km_from_centroid = sf::st_distance(geometry, 
                                          center_point, 
                                          by_element = TRUE) / 1000) %>%
  collect


dec_df <- dist_decay_df %>%
  ungroup %>%
  filter(year != max(year)) %>%
  left_join(bbs_species) %>%
  group_by(route_id, english) %>%
  summarize(km_from_centroid = mean(km_from_centroid), 
            phi = mean(phi)) %>%
  group_by(english) %>%
  # slope is decrease in phi per 1000 km
  summarize(m = list(lm(phi ~ km_from_centroid)), 
            phid_cor = m[[1]]$coef["km_from_centroid"], 
            lo = confint(m[[1]])['km_from_centroid', '2.5 %'], 
            hi = confint(m[[1]])['km_from_centroid', '97.5 %']) %>%
  left_join(dist_decay_df %>%
              ungroup %>%
              left_join(bbs_species) %>%
              filter(z_mle == 1) %>%
              count(english, sp.bbs)) %>%
  left_join(as_tibble(centroid_pts) %>%
              group_by(sp.bbs) %>%
              summarize(mean_bbs_cent_dist = mean(overall_bbs_centroid_dist))) %>%
  mutate(english = reorder(english, phid_cor))

to_label <- dec_df %>%
  top_n(1, phid_cor) %>%
  bind_rows(top_n(dec_df, 1, -phid_cor))


pt_alpha <- .3
pt_size <- .5
dist_cor_plot <- dec_df %>%
  ggplot(aes(n, phid_cor)) + 
  geom_hline(yintercept = 0, linetype = 'dashed') + 
  geom_point(alpha = pt_alpha, size = pt_size) + 
  theme_minimal() + 
  xlab("Total occurrences over all years") + 
  ylab(expression(paste(d["c"], " coefficient"))) + 
  ggtitle("(c)") + 
  theme(panel.grid.minor = element_blank()) + 
  geom_point(data = to_label, size = 1) + 
  geom_text_repel(aes(label = english), data = to_label)
dist_cor_plot

dec_df %>%
  select(-m) %>%
  write_csv('out/dec_df.csv')

p2 <- dist_decay_df %>%
  ungroup %>%
  left_join(bbs_species) %>%
  left_join(dec_df) %>%
  filter(english %in% to_label$english) %>%
  group_by(route_id, english) %>%
  summarize(km_from_centroid = mean(km_from_centroid), 
            phi = mean(phi),
            phid_cor = unique(phid_cor)) %>% 
  mutate(english = factor(english, levels = levels(dec_df$english))) %>%
  ggplot(aes(km_from_centroid, phi)) + 
  geom_point(alpha = pt_alpha, size = pt_size) + 
  facet_wrap(~reorder(english, -phid_cor), nrow = 2) + 
  xlab(expression(paste("Kilometers from range centroid (", d["c"], ")"))) + 
  ylab(expression(paste("Persistence probability (", phi, ")"))) + 
  theme_minimal() + 
  theme(panel.grid.minor = element_blank(), 
        axis.text.x = element_text(angle = 35)) + 
  ggtitle("(a)")
p2

p3 <- z_mles %>%
  filter(z_mle == 1, year != max(year)) %>%
  left_join(bbs_species) %>%
  filter(english %in% to_label$english) %>%
  left_join(bbs_routes) %>%
  group_by(Latitude, Longitude, english) %>%
  summarize(phi = mean(phi)) %>%
  st_as_sf(coords = c("Longitude", "Latitude"), 
           crs = 4326, agr = "constant") %>%
  st_transform(epsg) %>%
  left_join(dec_df) %>%
  mutate(english = reorder(english, -phid_cor)) %>%
  group_by(english) %>%
  mutate(frac_p = phi / max(phi)) %>%
  ggplot() + 
  geom_sf(data = ecoregions, 
          fill = 'white',  
          size =.1, alpha = .9) +
  geom_sf(data = routes_sf %>%
            filter(route_id %in% routes_surveyed_most_years$route_id), 
          alpha = .1, size = .1) +
  geom_sf(aes(color = phi), size = .1) +
  geom_sf(data = centroid_pts %>%
                  left_join(bbs_species) %>%
                  filter(english %in% to_label$english) %>%
                  mutate(english = factor(english, levels = levels(dec_df$english))) %>%
                  group_by(english) %>%
                  summarize() %>%
                  st_cast("LINESTRING"), 
          size = 1) +
  scale_color_viridis_c(option = "C", 
                        expression(paste(phi))) + 
  theme_minimal() + 
  facet_wrap(~english, nrow = 2) + 
  ggtitle("(b)")
p3


persist_dist_plot <- (p2 | p3) / dist_cor_plot + plot_layout(heights = c(1, .7))
persist_dist_plot

persist_dist_plot %>%
  {
    ggsave(filename = 'fig/persist-dist-plot.jpg', plot = ., 
           width = 5, height = 6, dpi = dpi)
    ggsave(filename = 'fig/persist-dist-plot.pdf', plot = ., 
           width = 5, height = 6)
  }

