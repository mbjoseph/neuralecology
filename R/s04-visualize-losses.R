library(tidyverse)

loss <- list.files("out/params", pattern = "loss_df", full.names = TRUE) %>%
  lapply(read_csv) %>%
  bind_rows %>%
  pivot_longer(ends_with("loss"))

loss_df <- loss %>%
  group_by(name, n) %>%
  summarize(valid_loss = sum(value))

label_df <- loss_df %>%
  ungroup %>%
  filter(n == 512) %>%
  mutate(name = case_when(
    .$name == "baseline_pt_loss" ~ "Point extraction", 
    .$name == "bestcase_loss" ~ "Best case", 
    .$name == "conv_loss" ~ "ConvHMM"
  ))

val_loss_plot <- loss_df %>%
  ggplot(aes(n, -valid_loss, color = name)) + 
  geom_line(alpha = .7) + 
  geom_point() + 
  scale_x_log10(breaks = c(2^(4:10))) + 
  xlab("Training set size") + 
  ylab("Validation set performance") + 
  theme_minimal() + 
  geom_text(data = label_df, 
            aes(label = name, 
                y = -valid_loss + 1500), color = "black") + 
  theme(legend.position = "none", 
        panel.grid.minor = element_blank()) + 
  scale_color_manual(values = c("darkorange1", "grey50", "darkorchid"))
val_loss_plot
ggsave("fig/convhmm-perf.pdf", val_loss_plot, width = 4, height = 3)
