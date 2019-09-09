
# Visualizing neural network weights --------------------------------------

library(tidyverse)
library(patchwork)
library(RColorBrewer)
library(viridis)
library(Rtsne)
library(vroom)
library(lsa)
library(pbapply)


theme_set(theme_minimal() + 
            theme(panel.grid.minor = element_blank()))

bbs_species <- read_csv('data/cleaned/bbs_species.csv') %>%
  mutate(sp_lower = tolower(english))

bbs_routes <- read_csv('data/cleaned/clean_routes.csv')


weight_files <- list.files(path = 'out/weights', full.names = TRUE)
en_files <- basename(weight_files) %>%
  gsub('_weights.csv', '', .) %>%
  gsub('_', ' ', .)

joined_df <- weight_files %>%
  lapply(read_csv) %>%
  bind_rows(.id = 'file') %>%
  mutate(file = parse_integer(file), 
         sp_lower = en_files[file]) %>%
  select(-file) %>%
  filter(par != 'p') %>%
  unite('par_h', c(par, h_dim)) %>%
  spread(par_h, value) %>%
  left_join(distinct(bbs_species, english, sp_lower)) %>%
  arrange(english) %>%
  select(-sp_lower)


# Cosine similarity analysis ----------------------------------------------

X <- joined_df %>%
  select(-english) %>%
  as.matrix
rownames(X) <- joined_df$english

sim <- cosine(t(X))

# cosine dissimilarity (distance) matrix
D_sim <- as.dist(1 - sim)

cosine_sim <- joined_df %>%
  select(english)

cosine_sim$nearest_species <- NA
cosine_sim$similarity <- NA
cosine_sim$next_nearest_species <- NA
cosine_sim$next_similarity <- NA
cosine_sim$farthest_species <- NA
cosine_sim$far_similarity <- NA

for (i in 1:nrow(cosine_sim)) {
  sim_vec <- sim[which(joined_df$english == cosine_sim$english[i]), ]
  names(sim_vec) <- joined_df$english
  
  nearest <- sort(sim_vec, decreasing = TRUE)[2]
  cosine_sim$similarity[i] <- nearest
  cosine_sim$nearest_species[i] <- names(nearest)
  
  next_nearest <- sort(sim_vec, decreasing = TRUE)[3]
  cosine_sim$next_nearest_species[i] <- names(next_nearest)
  cosine_sim$next_similarity[i] <- next_nearest
  
  farthest <- sort(sim_vec)[1]
  cosine_sim$far_similarity[i] <- farthest
  cosine_sim$farthest_species[i] <- names(farthest)
}

write_csv(cosine_sim, 'out/cosine_sim.csv')


make_species_plot <- function(sp1, sp2, legend = TRUE) {

  sp1_df <- sp1 %>%
    paste0(., '_finalnet.csv') %>%
    file.path('out', .) %>%
    read_csv %>%
    distinct(sp.bbs) %>%
    unlist %>%
    paste0('.csv') %>%
    file.path('out', 'q_dfs', .) %>%
    read_csv %>%
    select(sp.bbs, route_id, year, psi) %>%
    mutate(sp_lower = gsub("_", " ", sp1)) %>%
    left_join(distinct(bbs_species, sp_lower, english)) %>%
    rename(sp1_psi = psi) %>%
    select(route_id, year, ends_with("psi"), english)
  
  sp2_df <- sp2 %>%
    paste0(., '_finalnet.csv') %>%
    file.path('out', .) %>%
    read_csv %>%
    distinct(sp.bbs) %>%
    unlist %>%
    paste0('.csv') %>%
    file.path('out', 'q_dfs', .) %>%
    read_csv %>%
    select(sp.bbs, route_id, year, psi) %>%
    mutate(sp_lower = gsub("_", " ", sp2)) %>%
    left_join(distinct(bbs_species, sp_lower, english)) %>%
    rename(sp2_psi = psi) %>%
    select(route_id, year, ends_with("psi"), english)
  
  sp1_name <- unique(sp1_df$english)
  sp2_name <- unique(sp2_df$english)
  
  l1_labels <- read_csv('data/cleaned/routes.csv') %>%
    distinct(route_id, L1_KEY) %>%
    mutate(level_1_ecoregion = tools::toTitleCase(tolower(L1_KEY)), 
           level_1_ecoregion = gsub("[0-9]", "", level_1_ecoregion), 
           level_1_ecoregion = trimws(level_1_ecoregion))
  l1_order <- count(l1_labels, level_1_ecoregion) %>% arrange(-n)
  l1_labels$level_1_ecoregion <- factor(l1_labels$level_1_ecoregion, 
                                        levels = l1_order$level_1_ecoregion)
  
  p <- sp1_df %>%
    select(-english) %>%
    full_join(select(sp2_df, -english)) %>%
    left_join(l1_labels) %>%
    left_join(read_csv('data/cleaned/clean_routes.csv')) %>%
    filter(group != 'test', level_1_ecoregion != 'Water') %>%
    arrange(level_1_ecoregion) %>%
    ggplot(aes(x = sp1_psi, y = sp2_psi, 
               group = route_id, color = level_1_ecoregion)) + 
    geom_path(alpha = .2, size = .5) + 
    xlab(substitute(paste(sp1_name), 
                    list(sp1_name = sp1_name))) + 
    ylab(substitute(paste(sp2_name), 
                    list(sp2_name = sp2_name))) + 
    scale_color_brewer(palette = 'Paired', '') + 
    guides(colour = guide_legend(override.aes = list(alpha = 1, size = 1))) + 
    ylim(0, 1) + 
    xlim(0, 1)
  if (!legend) {
    p <- p + theme(legend.position = 'none')
  }
  p
}

make_pair_plot <- function(ref_sp, legend = TRUE, title = "") {
  cosine_df <- cosine_sim %>%
    filter(english == ref_sp)
  
  focal_lower <- tolower(ref_sp) %>%
    gsub(' ', '_', .)
  nearest_lower <- tolower(cosine_df$nearest_species) %>%
    gsub(' ', '_', .)
  next_nearest_lower <- tolower(cosine_df$next_nearest_species) %>%
    gsub(' ', '_', .)
  farthest_lower <- tolower(cosine_df$farthest_species) %>%
    gsub(' ', '_', .)
  
  p_near <-  make_species_plot(sp1 = nearest_lower, sp2 = focal_lower, 
                               legend = FALSE) +
    ggtitle(title) + 
    annotate(geom = "text", x = 0, y = 1, hjust = 0, vjust = 1,
             label = paste("Similarity:", round(cosine_df$similarity, 3)))
  
  p_far <- make_species_plot(sp1 = farthest_lower, sp2 = focal_lower, 
                             legend = legend) + 
    theme(axis.text.y = element_blank()) + 
    ylab("") + 
    annotate(geom = "text", x = 1, y = 1, hjust = 1, vjust = 1,
             label = paste("Similarity:", round(cosine_df$far_similarity, 3)))
  
  p <- p_near + p_far
  p
}

p <- make_pair_plot("Mourning Dove", legend = FALSE, title = "(a)") / 
  make_pair_plot("Eurasian Collared-Dove", title = "(b)") / 
  make_pair_plot("Wilson's Warbler", legend = FALSE, title = "(c)")
p


r_dpi <- 150


pwidth <- 9
pheight <- 6
p %>%
  {
    ggsave(filename = 'fig/occupancy_scatter.jpg', plot = ., 
           width = pwidth, height = pheight,
           dpi = r_dpi);
  }



# Parse route embeddings --------------------------------------------------

route_embeddings <- vroom('out/route_embeddings.csv')
route_embeddings$route_id <- vroom('out/american_robin_finalnet.csv')$route_id[route_embeddings$row_idx + 1]

psi0_embeddings <- route_embeddings %>%
  filter(par == 'psi0') %>%
  select(-t, -row_idx) %>%
  unite(par_h_dim, par, h_dim) %>%
  spread(par_h_dim, value)

embed_wide <- route_embeddings %>%
  filter(par != 'psi0') %>%
  select(-row_idx) %>%
  unite(par_h_dim, par, h_dim, t) %>%
  spread(par_h_dim, value) %>%
  left_join(read_csv('data/cleaned/clean_routes.csv')) %>%
  arrange(Longitude, Latitude) %>%
  left_join(psi0_embeddings) %>%
  left_join(distinct(bbs_routes, route_id, group)) %>%
  filter(group != 'test')

unique_embed <- embed_wide %>%
  select(starts_with('gamma'), starts_with('phi_'), 
         starts_with('psi0_'), starts_with('p_')) %>%
  distinct

set.seed(0)
r_tsne <- unique_embed %>%
  Rtsne(verbose = TRUE, pca = FALSE, 
        num_threads = parallel::detectCores())

colnames(r_tsne$Y) <- c("d1", "d2")
raw_tsne_df <- as_tibble(r_tsne$Y) 
tsne_df <- unique_embed %>%
  bind_cols(raw_tsne_df) %>%
  right_join(embed_wide) %>% 
  left_join(vroom('data/cleaned/clean_routes.csv') %>%
              distinct(route_id, Longitude, Latitude, L1_KEY, L2_KEY)) %>%
  mutate(level_1_ecoregion = tools::toTitleCase(tolower(L1_KEY)), 
         level_1_ecoregion = gsub("[0-9]", "", level_1_ecoregion), 
         level_1_ecoregion = trimws(level_1_ecoregion))

pt_size <- .7
route_tsne_p1 <- tsne_df %>%
  filter(!(level_1_ecoregion %in% c('Hudson Plain', 'Water'))) %>%
  ggplot(aes(d1, d2, color = level_1_ecoregion)) + 
  geom_point(size = pt_size * 1.5, alpha = .5) +
  scale_color_brewer(palette = 'Paired', '') + 
  xlab("t-SNE dimension 1") + 
  ylab("t-SNE dimension 2") + 
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 1.5))) + 
  theme(legend.position = 'left')
route_tsne_p2 <- tsne_df %>%
  select(d1, d2, PC1, starts_with("c_")) %>%
  gather(var, value, -d1, -d2) %>%
  group_by(var) %>%
  mutate(value = c(scale(value))) %>%
  ungroup %>%
  mutate(var = case_when(
           var == 'c_elevation' ~ 'Elevation', 
           var == 'c_lat' ~ 'Latitude', 
           var == 'c_lon' ~ 'Longitude',
           var == 'c_road_den' ~ 'Road density',
           var == 'c_dist_shore' ~ 'Distance from coast',
           TRUE ~ var
         )) %>%
  ggplot(aes(d1, d2, color = value)) + 
  geom_point(size = pt_size, alpha = .5) +
  scale_color_gradient2("Z-score", mid = 'grey95') + 
  xlab("") + 
  ylab("") + 
  facet_wrap(~var)
route_tsne_plot <- route_tsne_p1 + ggtitle("(a)") + 
  route_tsne_p2 + ggtitle("(b)") + 
  plot_layout(nrow = 1, widths = c(1, 1.5))
route_tsne_plot
tsne_w <- 11
tsne_h <- 3.5
route_tsne_plot  %>%
  {
    ggsave(filename = 'fig/route_tsne.jpg', plot = ., 
           width = tsne_w, height = tsne_h,
           dpi = r_dpi);
  }

