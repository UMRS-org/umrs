library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------

df_raw <- read.csv("log/knn_xp.csv", na.strings = c("None", "NA", ""))

# Outliers: server_time_s > 0.4s or client_real_s > 5s are timeouts / cold starts
df <- df_raw %>%
  filter(!is.na(server_time_s), !is.na(client_real_s)) %>%
  filter(server_time_s < 0.4, client_real_s < 5)

cat(sprintf("Rows kept: %d / %d (%.0f%% after outlier removal)\n",
            nrow(df), nrow(df_raw), 100 * nrow(df) / nrow(df_raw)))

# k as ordered factor for colour scale
df$k_f <- factor(df$k, levels = sort(unique(df$k)))

# ---------------------------------------------------------------------------
# Aggregate: mean ± sd over reps
# ---------------------------------------------------------------------------

agg <- df %>%
  group_by(k, n, k_f) %>%
  summarise(
    server_mean = mean(server_time_s) * 1000,   # → ms
    server_sd   = sd(server_time_s)  * 1000,
    client_mean = mean(client_real_s) * 1000,
    client_sd   = sd(client_real_s)  * 1000,
    nrep        = n(),
    .groups = "drop"
  )

# ---------------------------------------------------------------------------
# Palette: log2 scale on k (1,2,4,...,1024) → 11 levels
# ---------------------------------------------------------------------------

k_levels  <- sort(unique(agg$k))
pal_blues <- colorRampPalette(c("#cfe2f3", "#08519c"))(length(k_levels))
names(pal_blues) <- as.character(k_levels)

n_levels  <- sort(unique(agg$n))
pal_reds  <- colorRampPalette(c("#fcbba1", "#67000d"))(length(n_levels))
names(pal_reds) <- as.character(n_levels)

theme_knn <- theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    legend.key.height = unit(0.45, "cm"),
    strip.background = element_rect(fill = "#f0f0f0")
  )

# ---------------------------------------------------------------------------
# Figure 1 – server time vs n  (one line per k)
# ---------------------------------------------------------------------------

p1 <- ggplot(agg, aes(x = n, y = server_mean, colour = factor(k),
                       ymin = server_mean - server_sd,
                       ymax = server_mean + server_sd)) +
  geom_ribbon(aes(fill = factor(k)), alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = pal_blues, name = "k") +
  scale_fill_manual(values = pal_blues,   name = "k") +
  scale_x_continuous(breaks = n_levels,
                     labels = label_number(scale = 1e-3, suffix = "k")) +
  labs(
    title    = "KNN server time vs bucket size",
    subtitle = "pgvector cosine search — mean ± sd over 10 reps (outliers removed)",
    x        = "Bucket size  n",
    y        = "Server time (ms)"
  ) +
  theme_knn

ggsave("log/knn_server_vs_n.pdf", p1, width = 9, height = 5)
ggsave("log/knn_server_vs_n.png", p1, width = 9, height = 5, dpi = 150)

# ---------------------------------------------------------------------------
# Figure 2 – server time vs k  (one line per n)
# ---------------------------------------------------------------------------

p2 <- ggplot(agg, aes(x = k, y = server_mean, colour = factor(n),
                       ymin = server_mean - server_sd,
                       ymax = server_mean + server_sd)) +
  geom_ribbon(aes(fill = factor(n)), alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  scale_x_log10(breaks = k_levels) +
  scale_colour_manual(values = pal_reds, name = "n") +
  scale_fill_manual(values = pal_reds,   name = "n") +
  labs(
    title    = "KNN server time vs k",
    subtitle = "pgvector cosine search — mean ± sd over 10 reps (outliers removed)",
    x        = "k  (log scale)",
    y        = "Server time (ms)"
  ) +
  theme_knn

ggsave("log/knn_server_vs_k.pdf", p2, width = 9, height = 5)
ggsave("log/knn_server_vs_k.png", p2, width = 9, height = 5, dpi = 150)

# ---------------------------------------------------------------------------
# Figure 3 – client time vs n  (one line per k)
# ---------------------------------------------------------------------------

p3 <- ggplot(agg, aes(x = n, y = client_mean, colour = factor(k),
                       ymin = client_mean - client_sd,
                       ymax = client_mean + client_sd)) +
  geom_ribbon(aes(fill = factor(k)), alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = pal_blues, name = "k") +
  scale_fill_manual(values = pal_blues,   name = "k") +
  scale_x_continuous(breaks = n_levels,
                     labels = label_number(scale = 1e-3, suffix = "k")) +
  labs(
    title    = "KNN client time vs bucket size",
    subtitle = "End-to-end through network — mean ± sd over 10 reps (outliers removed)",
    x        = "Bucket size  n",
    y        = "Client time (ms)"
  ) +
  theme_knn

ggsave("log/knn_client_vs_n.pdf", p3, width = 9, height = 5)
ggsave("log/knn_client_vs_n.png", p3, width = 9, height = 5, dpi = 150)

# ---------------------------------------------------------------------------
# Figure 4 – client time vs k  (one line per n)
# ---------------------------------------------------------------------------

p4 <- ggplot(agg, aes(x = k, y = client_mean, colour = factor(n),
                       ymin = client_mean - client_sd,
                       ymax = client_mean + client_sd)) +
  geom_ribbon(aes(fill = factor(n)), alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  scale_x_log10(breaks = k_levels) +
  scale_colour_manual(values = pal_reds, name = "n") +
  scale_fill_manual(values = pal_reds,   name = "n") +
  labs(
    title    = "KNN client time vs k",
    subtitle = "End-to-end through network — mean ± sd over 10 reps (outliers removed)",
    x        = "k  (log scale)",
    y        = "Client time (ms)"
  ) +
  theme_knn

ggsave("log/knn_client_vs_k.pdf", p4, width = 9, height = 5)
ggsave("log/knn_client_vs_k.png", p4, width = 9, height = 5, dpi = 150)

# ---------------------------------------------------------------------------
# Figure 5 – server vs client time scatter (overhead = client - server)
# ---------------------------------------------------------------------------

agg_long <- agg %>%
  mutate(overhead_mean = client_mean - server_mean)

p5 <- ggplot(agg_long, aes(x = n, y = overhead_mean, colour = factor(k))) +
  geom_line(linewidth = 0.7, alpha = 0.8) +
  geom_point(size = 1.6) +
  scale_colour_manual(values = pal_blues, name = "k") +
  scale_x_continuous(breaks = n_levels,
                     labels = label_number(scale = 1e-3, suffix = "k")) +
  labs(
    title    = "Network overhead (client − server)",
    subtitle = "Dominated by ZMQ + serialisation, independent of n and k",
    x        = "Bucket size  n",
    y        = "Overhead (ms)"
  ) +
  theme_knn

ggsave("log/knn_overhead_vs_n.pdf", p5, width = 9, height = 5)
ggsave("log/knn_overhead_vs_n.png", p5, width = 9, height = 5, dpi = 150)

# ---------------------------------------------------------------------------
# Summary table printed to console
# ---------------------------------------------------------------------------

cat("\n--- Mean server time (ms) by n and k ---\n")
agg %>%
  select(n, k, server_mean) %>%
  pivot_wider(names_from = k, values_from = server_mean,
              names_prefix = "k=") %>%
  arrange(n) %>%
  print(n = Inf)

cat("\n--- Mean client time (ms) by n and k ---\n")
agg %>%
  select(n, k, client_mean) %>%
  pivot_wider(names_from = k, values_from = client_mean,
              names_prefix = "k=") %>%
  arrange(n) %>%
  print(n = Inf)

cat("\nPlots saved in log/\n")
