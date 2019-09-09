library(tidyverse)
library(assertthat)
library(sf)
library(vroom)
library(parallel)
library(pbapply)

counts <- vroom('data/bbs_aggregated/bird.csv') %>%
  rename(sp.bbs = AOU) %>%
  mutate(route_id = paste(sprintf('%02d', statenum), 
                          sprintf('%03d', Route), 
                          sep = '_'))

routes <- read_csv('data/bbs_aggregated/route.csv') %>%
  filter(Year >= 1997) %>%
  mutate(StartTemp = parse_number(StartTemp), 
         EndTemp = parse_number(EndTemp), 
         route_id = paste(sprintf('%02d', statenum), 
                          sprintf('%03d', Route), 
                          sep = '_'), 
         TempScale = ifelse(TempScale == 'f', 'F', TempScale), 
         TempScale = ifelse(TempScale == 'c', 'C', TempScale),
         # based on visual inspection, NULL and NA tempscales are F
         TempScale = ifelse(TempScale == "NULL", 'F', TempScale),
         TempScale = ifelse(is.na(TempScale), 'F', TempScale),
         # there are a lot of records with start/end temp 0, 0
         StartTemp = ifelse(StartTemp == 0 & TempScale == 'F', NA, StartTemp),
         EndTemp = ifelse(EndTemp == 0 & TempScale == 'F', NA, EndTemp),
         StartTemp = ifelse(TempScale == 'F',
                            (StartTemp - 32) * 5/9, StartTemp),
         EndTemp = ifelse(TempScale == 'F',
                          (EndTemp - 32) * 5 / 9, EndTemp), 
         new_temp_scale = 'C', 
         # remove crazy values that are obviously wrong
         EndTemp = ifelse(EndTemp > 55, NA, EndTemp), 
         EndTemp = ifelse(EndTemp < -5, NA, EndTemp), 
         StartTemp = ifelse(StartTemp > 55, NA, StartTemp)
         )


routes %>%
  ggplot(aes(StartTemp, EndTemp)) + 
  geom_point(alpha = .1) + 
  facet_wrap(~TempScale)

# ensure that all routes in the count data have route-level data
counts <- counts %>%
  filter(route_id %in% routes$route_id, 
         RouteDataID %in% routes$RouteDataID)

assert_that(all(counts$route_id %in% routes$route_id))
assert_that(all(counts$RouteDataID %in% routes$RouteDataID))

species <- read_csv('data/bbs_aggregated/species.csv')

rm_sp <- species %>%
  filter(grepl('unid', english, ignore.case = TRUE) | grepl('hybrid', english))

counts <- counts %>%
  filter(!(sp.bbs %in% rm_sp$sp.bbs))

# compute the number of stops where each species was seen
counts$y <- counts %>%
  select(starts_with('Stop')) %>%
  as.matrix %>%
  `!=`(., 0) %>%
  rowSums


count_combos <- counts %>%
  select(RouteDataID, sp.bbs, y) %>%
  complete(sp.bbs, RouteDataID, fill = list(y = 0)) %>%
  left_join(select(routes, RouteDataID, Year, route_id)) %>%
  arrange(route_id, Year, RouteDataID) %>%
  filter(Year >= 1997)

wide_counts <- count_combos %>%
  select(-RouteDataID) %>%
  spread(Year, y)

sp_to_plot <- species %>%
  filter(english == 'Eurasian Collared-Dove') %>%
  left_join(count_combos) %>%
  left_join(routes)

sp_to_plot %>%
  ggplot(aes(Longitude, Latitude, color = y)) + 
  geom_point(alpha = .02) + 
  geom_point(data = filter(sp_to_plot, y > 0)) +
  facet_wrap(~Year) + 
  coord_equal() + 
  theme_minimal() + 
  scale_color_viridis_c() +
  theme(panel.grid.minor = element_blank()) + 
  ggtitle(unique(sp_to_plot$english))

# all route data ids in the count data are represented in route data
assert_that(all(counts$RouteDataID %in% routes$RouteDataID))

dir.create('data/cleaned', recursive = TRUE)
write_csv(wide_counts, 'data/cleaned/bbs_counts.csv')
write_csv(species, 'data/cleaned/bbs_species.csv')
write_csv(routes, 'data/cleaned/bbs_routes.csv')

