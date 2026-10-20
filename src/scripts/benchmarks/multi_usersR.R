library(ggplot2)

# mac EJ
N    <- c(1, 2, 4, 8, 16, 32, 64, 80, 100)
Time <- c(0.960, 1.503, 2.482, 4.969, 10.377, 21.464, 38.548, 39.553, 43.713)

# ai-h100
N <- c(1, 2, 4, 8, 16, 32, 60, 80, 100)

Time <- c(
  
  2.190,
  
  4.504,
  
  9.350,
  
  19.761,
  
  40.392,
  
  77.012,
  
  138.209,
  
  174.176,
  
  198.937
  
)

# launch dir h100
N    <- c(  1,   2,   4,     8,       16,   32,      64,     128)
Time <- c(  3,   4,   5, 20.229,      49,   70, 153.847, 289.965)


throughput_1 <- 1 / Time[1]
speedup      <-  Time / Time[1]
ideal        <- N / N[1]

df <- data.frame(
  N       = N,
  speedup = speedup,
  ideal   = ideal
)



p <- ggplot(df, aes(x = N)) +
  geom_line(aes(y = ideal, color = "Ideal (linear)"),
            linetype = "dashed", linewidth = 1) +
  geom_line(aes(y = speedup, color = "Measured"),
            linewidth = 1) +
  geom_point(aes(y = speedup, color = "Measured"),
             size = 3) +
  scale_color_manual(values = c("Ideal (linear)" = "red",
                                "Measured"       = "steelblue")) +
  scale_x_continuous(breaks = N) +
  labs(
    title   = "Throughput Speedup vs Number of Concurrent Clients",
    x       = "Number of concurrent clients (N)",
    y       = "Throughput speedup",
    color   = NULL
  ) +
  theme_bw(base_size = 14) +
  theme(
    legend.position   = "topleft",
    panel.grid.minor  = element_blank()
  )

print(p)


# Table 1 — raw data
bench <- data.frame(
  n_users    = c(1, 2, 4, 8, 16, 32, 64, 128),
  duration_s = c(5.37, 6.12, 7.53, 12.09, 20.28, 38.13, 118.05, 262.39)
)

# Table 2 — derived metrics (32 objects per user)
bench$total_images  <- bench$n_users * 32
bench$throughput    <- bench$total_images / bench$duration_s          # img/s
bench$serial_s      <- bench$n_users * bench$duration_s[1]            # temps série théorique
bench$speedup       <- bench$serial_s / bench$duration_s              # accélération vs série
bench$efficiency    <- bench$speedup / bench$n_users                  # efficacité parallèle [0-1]



library(ggplot2)
library(tidyr)

# ── Data ──────────────────────────────────────────────────────────────────────
bench <- data.frame(
  objects_per_users=c(32, 32, 32, 32, 32, 32, 32, 32),
  n_users = c(1, 2, 4, 8, 16, 32, 64, 128),
  batch_s = c(5.37, 6.12, 7.53, 12.09, 20.28, 38.13, 118.05, 262.39)
)

# novelle version: 
bench <- data.frame(
  objects_per_users = c(32, 32, 32, 32, 32, 32, 32, 32),
  n_users = c(1, 2, 4, 8, 16, 32, 64, 128),
  batch_s = c(6.203, 7.190, 8.362, 13.742, 22.118, 41.103, 103.287, 328.055)
)

bench$serial_s <- bench$n_users * bench$batch_s[1]   # N × t(1 user)
bench$saved_s  <- bench$serial_s - bench$batch_s
bench$speedup  <- bench$serial_s / bench$batch_s
bench$n_label  <- factor(bench$n_users, levels = bench$n_users)

# ── Reshape: stacked bars (batch bottom, savings top = serial) ─────────────────
long <- pivot_longer(bench, c(batch_s, saved_s),
                     names_to = "part", values_to = "s")
long$part <- factor(long$part,
                    levels = c("saved_s", "batch_s"),
                    labels = c("Time saved by batching", "Batch time"))

# ── Plot ──────────────────────────────────────────────────────────────────────
ggplot(long, aes(x = n_label, y = s, fill = part)) +
  geom_col(width = 0.72) +
  
  # Speedup label on top of each bar; peak (N=32) in gold
  geom_text(
    data        = bench,
    mapping     = aes(x      = n_label,
                      y      = serial_s,
                      label  = sprintf("×%.2f", speedup),
                      colour = n_users == 32),
    inherit.aes = FALSE,
    vjust = -0.35, fontface = "bold", size = 3.7
  ) +
  
  # Arrow pointing at the N=32 peak label
  annotate("segment",
           x = 6.55, xend = 6.08, y = 228, yend = 188,
           arrow  = arrow(length = unit(0.18, "cm"), type = "closed"),
           colour = "#d4a000", linewidth = 0.8) +
  annotate("text",
           x = 6.62, y = 235,
           label = "GPU sweet spot", hjust = 0,
           colour = "#d4a000", size = 3.4, fontface = "bold") +
  
  scale_fill_manual(values = c(
    "Time saved by batching" = "#d96828",
    "Batch time"             = "#10c890"
  )) +
  scale_colour_manual(
    values = c("TRUE" = "#d4a000", "FALSE" = "#444444"),
    guide  = "none"
  ) +
  #scale_y_log10()+
  scale_y_continuous(expand = expansion(mult = c(0, 0.14)),
                     labels = \(x) paste0(x, " s")) +
  
  labs(
    title    = "GPU batch inference: up to 4.51× faster than serial",
    subtitle = paste0(
      "Full bar = serial cost (N × 5.37 s).  ",
      "Green = actual batch time.  Orange = time saved by batching."
    ),
    x    = "Concurrent users (N)",
    y    = "Time",
    fill = NULL
  ) +
  
  theme_minimal(base_size = 13) +
  theme(
    plot.title         = element_text(face = "bold", size = 14),
    plot.subtitle      = element_text(colour = "grey50", size = 10,
                                      margin = margin(b = 10)),
    legend.position    = "top",
    legend.text        = element_text(size = 10),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank()
  )

ggsave("log/batch_speedup.pdf", width = 9, height = 5.5)
message("Saved: batch_speedup.pdf / batch_speedup.png")

library(ggplot2)
library(tidyr)

bench <- data.frame(
  n_users = c(1, 2, 4, 8, 16, 32, 64, 128),
  batch_s = c(6.203, 7.190, 8.362, 13.742, 22.118, 41.103, 103.287, 328.055)
)

bench$serial_s <- bench$n_users * bench$batch_s[1]   # extrapolation
bench$speedup  <- bench$serial_s / bench$batch_s
bench$n_label  <- factor(bench$n_users, levels = bench$n_users)

long <- pivot_longer(bench, c(batch_s, serial_s),
                     names_to  = "series",
                     values_to = "s")
long$series <- factor(long$series,
                      levels = c("serial_s", "batch_s"),
                      labels = c("Sequential baseline (N \u00d7 t\u2081, extrapolated)",
                                 "Batch (measured)"))

ggplot(long, aes(x = n_label, y = s, fill = series)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  
  # speedup label above the taller bar (serial_s)
  geom_text(
    data    = bench,
    mapping = aes(x      = n_label,
                  y      = serial_s,
                  label  = sprintf("\u00d7%.2f", speedup),
                  colour = n_users == 32),
    inherit.aes = FALSE,
    position = position_dodge(width = 0.75),
    vjust = -0.4, fontface = "bold", size = 3.6
  ) +
  
  annotate("segment",
           x = 6.45, xend = 6.10, y = 228, yend = 195,
           arrow  = arrow(length = unit(0.18, "cm"), type = "closed"),
           colour = "#d4a000", linewidth = 0.8) +
  annotate("text",
           x = 6.52, y = 234,
           label = "GPU sweet spot", hjust = 0,
           colour = "#d4a000", size = 3.4, fontface = "bold") +
  
  scale_fill_manual(values = c(
    "Sequential baseline (N \u00d7 t\u2081, extrapolated)" = "#d96828",
    "Batch (measured)"                                     = "#10c890"
  )) +
  scale_colour_manual(
    values = c("TRUE" = "#d4a000", "FALSE" = "#444444"),
    guide  = "none"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.16)),
                     labels = \(x) paste0(x, " s")) +
  
  labs(
    title    = "GPU batch inference: up to 4.51\u00d7 faster than sequential",
    subtitle = "Orange = sequential cost extrapolated as N \u00d7 t(1 user). Green = actual measured batch time.",
    x    = "Concurrent users (N)",
    y    = "Time (s)",
    fill = NULL
  ) +
  
  theme_minimal(base_size = 13) +
  theme(
    plot.title         = element_text(face = "bold", size = 14),
    plot.subtitle      = element_text(colour = "grey50", size = 10,
                                      margin = margin(b = 10)),
    legend.position    = "top",
    legend.text        = element_text(size = 10),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank()
  )

ggsave("log/batch_speedup_side.pdf", width = 9, height = 5.5)

message("Saved: log/batch_speedup_side.pdf")