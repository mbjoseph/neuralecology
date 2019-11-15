library(tidyverse)

loss <- list.files("out/params", pattern = "loss.csv", full.names = TRUE) %>%
  lapply(read_csv) %>%
  bind_rows %>%
  group_by(model, n_train, group) %>%
  mutate(iter = 1:n() / n()) %>%
  filter(iter > .1) 


loss %>%
  ggplot(aes(iter, loss, color = model)) + 
  geom_line(alpha = .7) + 
  facet_grid(group~n_train) + 
  scale_y_log10()
