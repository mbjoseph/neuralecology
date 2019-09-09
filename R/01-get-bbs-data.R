# Get data

library(bbsBayes)
library(tidyverse)
library(here)

# note this requires mbjoseph/bbsBayes@noninteractive from GitHub
# remotes::install_github("mbjoseph/bbsBayes@noninteractive")
fetch_bbs_data(level = 'stop') 

load(list.files("~/.local/share/bbsBayes", full.names = TRUE))

historical_dir <- here('data', 'bbs_aggregated')
dir.create(historical_dir, recursive = TRUE, showWarnings = FALSE)
out_files <- historical_dir %>%
  file.path(names(bbs_data)) %>%
  paste0('.csv')
for (i in seq_along(bbs_data)) {
  write_csv(bbs_data[[i]], out_files[i])
}
