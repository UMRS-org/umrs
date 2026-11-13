library(ggplot2)
library(dplyr)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

bench <- data.frame(
  n_users          = c(1, 2, 4, 8, 16, 32, 64, 128),
  objects_per_user = 32,
  batch_s          = c(5.37, 6.12, 7.53, 12.09, 20.28, 38.13, 118.05, 262.39)
)

# avec chnunking
bench <- data.frame(
  objects_per_user = 32,
  n_users = c(1, 2, 4, 8, 16, 32, 64, 128),
  batch_s = c(6.203, 7.190, 8.362, 13.742, 22.118, 41.103, 103.287, 328.055)
)

bench <- bench %>%
  mutate(
    n_objects      = n_users * objects_per_user,
    throughput     = n_objects / batch_s,                        # actual obj/s
    throughput_ideal = n_users * (objects_per_user / batch_s[1]) # linear scaling from 1 user
  )

peak_row <- bench[which.max(bench$throughput), ]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

ggplot(bench, aes(x = n_users)) +

  # Actual throughput
  geom_line(aes(y = throughput), colour = "#2166ac", linewidth = 1.1) +
  geom_point(aes(y = throughput,
                 colour = n_users == peak_row$n_users,
                 size   = n_users == peak_row$n_users)) +

  # Throughput labels on each point
  geom_text(aes(y     = throughput,
                label = sprintf("%.1f", throughput),
                colour = n_users == peak_row$n_users),
            vjust = -0.8, fontface = "bold", size = 3.4) +

  # Annotation: peak
  annotate("segment",
           x = peak_row$n_users + 6, xend = peak_row$n_users + 1,
           y = peak_row$throughput - 1,
           yend = peak_row$throughput - 0.3,
           arrow  = arrow(length = unit(0.18, "cm"), type = "closed"),
           colour = "#d4a000", linewidth = 0.8) +
  annotate("text",
           x = peak_row$n_users + 7, y = peak_row$throughput - 1.5,
           label = "GPU sweet spot", hjust = 0,
           colour = "#d4a000", size = 3.4, fontface = "bold") +


  scale_x_continuous(
    breaks = bench$n_users,
    labels = bench$n_users,
    trans  = "log2"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
  scale_colour_manual(values = c("TRUE" = "#d4a000", "FALSE" = "#2166ac"),
                      guide = "none") +
  scale_size_manual(values   = c("TRUE" = 4.5,       "FALSE" = 2.8),
                    guide = "none") +

  labs(
    title    = "Ingestion throughput vs concurrent users — H100, ResNet-50",
    subtitle = sprintf(
      "Each user submits %d objects.  Throughput peaks at %d users (%.1f obj/s), then drops as GPU saturates.",
      bench$objects_per_user[1], peak_row$n_users, peak_row$throughput
    ),
    x = "Concurrent users (log₂ scale)",
    y = "Throughput (objects / s)"
  ) +

  theme_bw(base_size = 13) +
  theme(
    plot.title         = element_text(face = "bold", size = 13),
    plot.subtitle      = element_text(colour = "grey40", size = 9.5,
                                      margin = margin(b = 10)),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank()
  )

ggsave("log/throughput_users.pdf", width = 9, height = 5.5)
message("Saved: log/throughput_users.pdf")
