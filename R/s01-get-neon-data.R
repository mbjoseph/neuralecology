library(data.table)
library(raster)
library(tidyverse)
library(neonUtilities)
library(geoNEON)
library(sf)

# Get NEON CHM and RGB data for the San Joaqin Experimental Range NEON site
# (adapted from https://www.neonscience.org/tree-heights-veg-structure-chm)
veglist <- loadByProduct(dpID="DP1.10098.001", site="SJER", package="basic", 
                         check.size = FALSE)
vegmap <- def.calc.geo.os(veglist$vst_mappingandtagging, 
                          "vst_mappingandtagging") %>%
  filter(!is.na(adjEasting)) %>%
  as_tibble

# Compute one coordinate per plot
plot_coords <- vegmap %>%
  group_by(plotID) %>%
  summarize(easting = mean(adjEasting, na.rm = TRUE), 
            northing = mean(adjNorthing, na.rm = TRUE))

coord_grid <- expand.grid(easting = seq(min(plot_coords$easting), 
                                        max(plot_coords$easting), 
                                        by = 100), 
                          northing = seq(min(plot_coords$northing), 
                                         max(plot_coords$northing), 
                                         by = 100)) %>%
  as_tibble

plot_coords %>%
  ggplot(aes(easting, northing)) + 
  geom_point() + 
  geom_point(data = coord_grid, color = 'red', size = .1)

# download the CHM data
byTileAOP(dpID="DP3.30015.001", 
          site="SJER", 
          year="2018", 
          easting=coord_grid$easting, 
          northing=coord_grid$northing,
          savepath="data", 
          check.size = FALSE)

# get the rgb imagery too
byTileAOP(dpID="DP3.30010.001", 
          site="SJER", 
          year="2018", 
          easting=coord_grid$easting, 
          northing=coord_grid$northing,
          savepath="data", 
          check.size = FALSE)

# Visualize the tile boundaries for the CHM data
chm_tiles <- list.files("data/DP3.30015.001/2018/FullSite/D17/2018_SJER_3/Metadata/DiscreteLidar/TileBoundary/shps", 
                        full.names = TRUE, pattern = "\\.shp$") %>%
  lapply(st_read) %>%
  data.table::rbindlist() %>%
  st_as_sf %>%
  st_zm
chm_tiles %>%
  ggplot() + 
  geom_sf()


chm_files <- list.files("data/DP3.30015.001/2018/FullSite/D17/2018_SJER_3/L3/DiscreteLidar/CanopyHeightModelGtif", 
                        full.names = TRUE, 
                        pattern = "\\.tif$")
rgb_files <- list.files("data/DP3.30010.001/2018/FullSite/D17/2018_SJER_3/L3/Camera/Mosaic", 
                        full.names = TRUE, 
                        pattern = "\\.tif$")


# Mosaic the chm
chm_list <- chm_files %>%
  lapply(raster)
chm_list$fun <- mean
chm_list$na.rm <- TRUE
chm_mosaic <- do.call(mosaic, chm_list)
writeRaster(chm_mosaic, 
            "out/chm_mosaic.tif", 
            overwrite = TRUE)

# Create an RGB mosaic (from stackoverflow)
# https://gis.stackexchange.com/questions/230553/merging-all-tiles-from-one-directory-using-gdal
system("gdalbuildvrt out/mosaic.vrt data/DP3.30010.001/2018/FullSite/D17/2018_SJER_3/L3/Camera/Mosaic/*.tif")
system("gdal_translate -of GTiff -co 'COMPRESS=JPEG' -co 'PHOTOMETRIC=YCBCR' -co 'TILED=YES' out/mosaic.vrt out/rgb_mosaic.tif")
rgb_mosaic <- stack("out/rgb_mosaic.tif")
plotRGB(rgb_mosaic, axes = TRUE, margins = TRUE)

