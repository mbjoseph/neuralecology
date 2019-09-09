library(tidyverse)
library(sf)
library(assertthat)
library(lubridate)
library(vroom)

routes <- read_csv('data/cleaned/routes.csv') %>%
  filter(!is.na(PC1), !is.na(L1_KEY), !is.na(elevation))

counts <- vroom('data/cleaned/bbs_counts.csv') %>%
  filter(route_id %in% routes$route_id)
species <- read_csv('data/cleaned/bbs_species.csv')
route_covariates <- read_csv('data/cleaned/bbs_routes.csv') %>%
  filter(route_id %in% routes$route_id)

long_route_covs <- route_covariates %>%
  select(route_id, Year, StartTemp, EndTemp, BCR)

# fewer than 1% of the start & end temp values are missing, so let's interpolate
mean(is.na(long_route_covs$StartTemp))
mean(is.na(long_route_covs$EndTemp))

long_route_covs <- long_route_covs %>%
  group_by(route_id) %>%
  mutate(StartTemp = ifelse(is.na(StartTemp), median(StartTemp), StartTemp), 
         EndTemp = ifelse(is.na(EndTemp), median(EndTemp), EndTemp)) %>%
  ungroup %>%
  # some route_ids have no temperature data at all, in those cases, group by
  # bird conservation region
  group_by(BCR, Year) %>%
  mutate(StartTemp = ifelse(is.na(StartTemp), 
                            median(StartTemp, na.rm = TRUE), 
                            StartTemp), 
         EndTemp = ifelse(is.na(EndTemp), 
                          median(EndTemp, na.rm = TRUE), 
                          EndTemp)) %>%
  ungroup %>%
  gather(which_temp, value, -route_id, -Year, - BCR) %>%
  mutate(route_id_year = paste(route_id, Year, sep = "_")) %>%
  arrange(route_id, Year, which_temp)

assert_that(!any(is.na(long_route_covs$mean_temp)))

long_route_covs[1:1000, ] %>%
  ggplot(aes(x = Year, y = value, color = which_temp)) + 
  geom_point() + 
  facet_wrap(~ route_id)


# Ensure that whenever we have bird detection data, we have survey temp. data
survey_counts_available <- counts %>%
  filter(sp.bbs == 10) %>% # choose an arbitrary species to check
  select(-sp.bbs) %>%
  gather(year, count, -route_id) %>%
  filter(!is.na(count)) %>%
  mutate(route_id_year = paste(route_id, year, sep = '_'))

assert_that(
  all(survey_counts_available$route_id_year %in% long_route_covs$route_id_year))


wind_df <- route_covariates %>%
  select(route_id, Year, StartWind, EndWind) %>%
  gather(var, value, -route_id, -Year) %>%
  mutate(value = value / max(value)) %>%
  unite(year_label, var, Year) %>%
  spread(year_label, value)


sky_df <-  route_covariates %>%
  select(route_id, Year, StartSky, EndSky) %>%
  gather(var, value, -route_id, -Year) %>%
  mutate(value = value / max(value)) %>%
  unite(year_label, var, Year) %>%
  spread(year_label, value)

duration_df <- route_covariates %>%
  select(route_id, Year, StartTime, EndTime) %>%
  mutate(StartTime = sprintf('%04d', StartTime), 
         EndTime = sprintf('%04d', EndTime), 
         pseudo_start_dt = lubridate::ymd_hms(paste0(Year, "-01-01 ", 
                                    substr(StartTime, 1, 2), ":",
                                    substr(StartTime, 3, 4), ":00")), 
         pseudo_end_dt = lubridate::ymd_hms(paste0(Year, "-01-01 ", 
                                                   substr(EndTime, 1, 2), ":",
                                                   substr(EndTime, 3, 4), ":00")), 
         duration = pseudo_end_dt - pseudo_start_dt, 
         # five of 46k records have unknown duration - plug in the median
         duration = ifelse(is.na(duration), median(duration, na.rm = TRUE), 
                           duration), 
         # three records have incorrect duration - plug in the median
         duration = ifelse(duration < 12, median(duration, na.rm = TRUE), 
                           duration)) %>%
  select(route_id, Year, duration) %>%
  mutate(duration = c(scale(duration)),
         year_label = paste0('duration_', Year)) %>%
  select(-Year) %>%
  spread(year_label, duration)
  
# detection covariates, they must also be in wide foramt
wide_route_covs <- long_route_covs %>%
  select(-route_id_year, -BCR) %>%
  mutate(value = c(scale(value))) %>%
  unite(year_label, which_temp, Year) %>%
  spread(year_label, value) %>%
  left_join(wind_df) %>%
  left_join(sky_df) %>%
  left_join(duration_df)


clean_routes <- routes %>%
  mutate(c_lat = c(scale(Latitude)), 
         c_lon = c(scale(Longitude)), 
         adj_elev = abs(min(elevation)) + elevation + 100,
         c_elevation = c(scale(log(adj_elev))), 
         c_dist_shore = c(scale(log(dist_shore))))

bbs <- counts %>%
  left_join(species) %>%
  left_join(wide_route_covs) %>%
  mutate(aou = paste0('aou_', aou))

assert_that(all(bbs$route_id %in% clean_routes$route_id))


# Generate summary stats about the size of the dataset
summary_df <- tibble(n_routes = length(unique(clean_routes$route_id)), 
                     n_species = length(unique(bbs$sp.bbs)))
summary_df <- bbs %>%
  left_join(clean_routes) %>%
  filter(sp.bbs == 10) %>%
  select(route_id, as.character(1997:2018)) %>%
  gather(year, y, -route_id) %>%
  summarize(n_surveys = sum(!is.na(y))) %>%
  bind_cols(summary_df)
summary_df <- summary_df %>%
  mutate(n_stops = n_surveys * 50, 
         n_detection_records = n_surveys * n_species)



# Write output files to disk ----------------------------------------------

summary_df %>%
  write_csv('data/cleaned/bbs-summary.csv')  

bbs %>%
  write_csv('data/cleaned/bbs.csv')

clean_routes %>%
  write_csv('data/cleaned/clean_routes.csv')
