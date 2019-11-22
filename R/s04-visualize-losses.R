library(tidyverse)

loss <- list.files("out/params", pattern = "loss_df", full.names = TRUE) %>%
  lapply(read_csv) %>%
  bind_rows %>%
  pivot_longer(ends_with("loss"))

loss %>%
  group_by(name, n) %>%
  summarize(valid_loss = sum(value)) %>%
  ggplot(aes(n, -valid_loss, color = name)) + 
  geom_line(alpha = .7) + 
  geom_point() + 
  scale_x_log10(breaks = c(2^(4:10))) + 
  xlab("Training set size") + 
  ylab("Validation log-likelihood") + 
  theme_minimal() + 
  annotate(geom = "text", x = 512, y = -58000, label = "Best case") + 
  annotate(geom = "text", x = 512, y = -62000, label = "ConvHMM") + 
  annotate(geom = "text", x = 512, y = -65500, label = "Point extraction") + 
  theme(legend.position = "none", 
        panel.grid.minor = element_blank()) + 
  scale_color_manual(values = c("darkorange1", "grey50", "darkorchid"))
