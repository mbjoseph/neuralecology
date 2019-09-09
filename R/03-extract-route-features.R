library(sf)
library(fasterize)
library(raster)
library(tidyverse)
library(RStoolbox)
library(elevatr)
library(rmapshaper)
library(rnaturalearth)

ecoregions <- st_read('data/NA_CEC_Eco_Level3.shp') %>%
  st_transform(4326)

# Extract the ecoregion data for BBS routes -----------------
routes <- read_csv('data/cleaned/bbs_routes.csv') %>%
  mutate(route_id = paste(sprintf('%02d', statenum), 
                   sprintf('%03d', Route), 
                   sep = '_'))

get_ecoregions <- function(sf, ecoregions) {
  intx <- st_intersects(sf, ecoregions)
  indices <- lapply(intx, function(x) {
    if (length(x) == 0) {
      x <- NA
    }
    x
  }) %>%
    unlist
  
  sf$L3_KEY <- ecoregions$NA_L3KEY[indices]
  sf$L2_KEY <- ecoregions$NA_L2KEY[indices]
  sf$L1_KEY <- ecoregions$NA_L1KEY[indices] 
  sf
}
  



# Get bioclim data --------------------------------------------------------

bioclim_raster <- 'data/bioclim/bioclim_pca.rds'
if (!file.exists(bioclim_raster)) {
  download.file('http://biogeo.ucdavis.edu/data/worldclim/v2.0/tif/base/wc2.0_5m_bio.zip', 
                'data/wc2.0_5m_bio.zip')
  dir.create('data/bioclim', showWarnings = FALSE)
  unzip('data/wc2.0_5m_bio.zip', exdir = 'data/bioclim')
  bioclim_files <- list.files(path = 'data/bioclim', pattern = '.tif', 
                              full.names = TRUE)
  bioclim <- stack(bioclim_files) 
  
  bioclim <- crop(bioclim, 
                  as(st_transform(ecoregions, crs(bioclim)), 'Spatial'))
  
  er_raster <- ecoregions %>%
    mutate(l3_int = as.numeric(factor(NA_L3KEY))) %>%
    st_transform(crs(bioclim)) %>%
    fasterize(bioclim[[1]], field = "l3_int", fun = "first")
  
  bioclim <- bioclim %>%
    mask(er_raster)

  bioclim_pca <- rasterPCA(bioclim, spca = TRUE)
  write_rds(bioclim_pca, bioclim_raster)
} else {
  bioclim_pca <- read_rds(bioclim_raster)
}

# first eight dimensions account for > 99% of the variance
summary(bioclim_pca$model)

routes_sf <- routes %>%
  distinct(route_id, Latitude, Longitude) %>%
  st_as_sf(coords = c('Longitude', 'Latitude'), 
           crs = 4326) %>%
  st_transform(st_crs(ecoregions)) %>%
  get_ecoregions(ecoregions) %>%
  filter(!is.na(L3_KEY)) # one route was NA

# extract PCs for BBS routes
bioclim_df <- routes_sf %>%
  st_transform(crs(bioclim_pca$map)) %>%
  raster::extract(bioclim_pca$map, .) %>%
  as_tibble %>%
  dplyr::select(PC1:PC8) %>%
  mutate(route_id = routes_sf$route_id)

# get elevation data
if (!file.exists('data/elevation.rds')) {
  elev <- get_elev_point(as(routes_sf, "Spatial"), src="aws", units="meters")
  write_rds(elev, 'data/elevation.rds')
} else {
  elev <- read_rds('data/elevation.rds')
}
elev_df <- as.data.frame(elev) %>%
  as_tibble() %>%
  dplyr::select(route_id, elevation)

# get grip road density data
roads_url <- 'http://geoservice.pbl.nl/download/opendata/GRIP4/GRIP4_density_total.zip'
if (!file.exists('data/grip4_total_dens_m_km2.asc')) {
  download.file(roads_url, destfile = file.path('data', basename(roads_url)))
  unzip(file.path('data', basename(roads_url)), exdir = 'data')
}
road_den <- raster('data/grip4_total_dens_m_km2.asc')
projection(road_den) <- st_crs(routes_sf)$proj4string
road_den <- projectRaster(road_den, crs = CRS(st_crs(routes_sf)$proj4string))
routes_sf$road_den <- raster::extract(road_den, as(routes_sf, "Spatial"), 
                                      buffer = 10000, fun = mean) %>%
  unlist
routes_sf$road_den <- ifelse(is.na(routes_sf$road_den), 
                             mean(routes_sf$road_den, na.rm = TRUE), 
                             routes_sf$road_den)
routes_sf$c_road_den <- c(scale(log(routes_sf$road_den + 1)))
plot(routes_sf['c_road_den'], pch = 19, cex = .8)


# Get distance to coastline data ------------------------------------------
eco_union <- ecoregions %>%
  summarize

simple_eco_union <- ms_simplify(eco_union, keep = 0.001) %>%
  st_transform(st_crs(routes_sf)) %>%
  st_cast('MULTILINESTRING')

plot(simple_eco_union)

routes_sf$dist_shore <- NA
pb <- txtProgressBar(max = nrow(routes_sf), style = 3)
for (j in seq_len(nrow(routes_sf))) {
  routes_sf$dist_shore[j] <- st_distance(routes_sf[j, ], simple_eco_union)
  setTxtProgressBar(pb, j)
}





# split data into partitions, blocking by level 3 ecoregion -----------------
set.seed(0)
routes_partition <- routes_sf %>%
  as_tibble() %>%
  distinct(L2_KEY) %>%
  mutate(group = sample(c('train', 'test', 'validation'), 
                        size = n(),
                        prob = c(1/3, 1/3, 1/3), 
                        replace = TRUE)) 

routes_sf <- routes_sf %>%
  left_join(routes_partition) %>%
  left_join(bioclim_df) %>%
  left_join(elev_df)

table(routes_sf$group)

plot(routes_sf, pch = 19, max.plot = 18, cex = .5)

routes_sf %>%
  as.data.frame %>%
  dplyr::select(-geometry) %>%
  as_tibble() %>%
  left_join(distinct(routes, route_id, Latitude, Longitude)) %>%
  write_csv('data/cleaned/routes.csv')

st_write(routes_sf, 'data/cleaned/routes.shp', delete_dsn = TRUE)

