library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df <- read.csv("log/h100_resnet.csv", check.names = TRUE)
# columns: task_id, user, n_batch, n_skipped, n_processed,
#          upload_s3.s., s3_check.s., server_exec.s., end_to_end.s.

names(df)[1:9] <- c("task_id","user","n_batch","n_skipped","n_processed",
               "upload_s3","s3_check","server_exec","end_to_end")

# end_to_end ≈ s3_check + server_exec (residual < 0.1% — negligible network overhead)
# decompose into two meaningful layers:
#   1. s3_check     (already-indexed pre-filter)
#   2. server_exec  (embedding computation + DB store)

# Per-object rates
df <- df %>%
  mutate(
    server_exec_ms_obj  = server_exec  / n_processed * 1000,
    end_to_end_ms_obj   = end_to_end   / n_processed * 1000,
    upload_s3_ms_obj    = upload_s3    / n_processed * 1000,
    # throughput: objects / second on each phase
    throughput_server   = n_processed / server_exec,
    throughput_e2e      = n_processed / end_to_end,
    throughput_upload   = n_processed / upload_s3
  )

# Average over repeated runs at the same n_processed
agg <- df %>%
  group_by(n_processed) %>%
  summarise(
    upload_s3   = mean(upload_s3),
    s3_check    = mean(s3_check),
    server_exec = mean(server_exec),
    end_to_end  = mean(end_to_end),
    server_exec_ms_obj  = mean(server_exec_ms_obj),
    end_to_end_ms_obj   = mean(end_to_end_ms_obj),
    upload_s3_ms_obj    = mean(upload_s3_ms_obj),
    throughput_server   = mean(throughput_server),
    throughput_e2e      = mean(throughput_e2e),
    throughput_upload   = mean(throughput_upload),
    nrep = n(),
    .groups = "drop"
  )

# ---------------------------------------------------------------------------
# Palette & theme
# ---------------------------------------------------------------------------

COL_SERVER  <- "#2166ac"   # blue  — server embedding + store
COL_S3CHK   <- "#92c5de"   # light blue — s3 check (pre-filter)
COL_OVERHEAD<- "#d1e5f0"   # very light — residual overhead
COL_UPLOAD  <- "#f4a582"   # orange — S3 upload (reference)

# Label formatter: 1 → "1", 100 → "100", 1000 → "1k", 7000 → "7k"
fmt_n <- function(x) ifelse(x >= 1000, paste0(x / 1000, "k"), as.character(x))

theme_lat <- theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        legend.position  = "bottom",
        legend.key.height = unit(0.4, "cm"))

# ---------------------------------------------------------------------------
# Figure 1 – stacked bar: end_to_end decomposed, + upload reference
# ---------------------------------------------------------------------------

# Pivot to long for stacking: s3_check (bottom) / server_exec (top)
stacked <- agg %>%
  select(n_processed, s3_check, server_exec) %>%
  pivot_longer(-n_processed,
               names_to  = "component",
               values_to = "seconds") %>%
  mutate(component = factor(component,
           levels = c("s3_check", "server_exec"),
           labels = c("S3 dedup check", "Server execution (embedding + store)")))

p1 <- ggplot(stacked, aes(x = factor(n_processed), y = seconds, fill = component)) +
  geom_col(width = 0.65) +
  # upload_s3 as a reference point (segment above bars)
  geom_point(data = agg,
             aes(x = factor(n_processed), y = upload_s3),
             inherit.aes = FALSE,
             shape = 18, size = 3, colour = COL_UPLOAD) +
  geom_line(data = agg,
            aes(x = factor(n_processed), y = upload_s3, group = 1),
            inherit.aes = FALSE,
            colour = COL_UPLOAD, linewidth = 0.8, linetype = "dashed") +
  annotate("text", x = 0.6, y = max(agg$upload_s3) * 0.92,
           label = "S3 upload", colour = COL_UPLOAD, hjust = 0, size = 3.5) +
  scale_x_discrete(labels = function(x) fmt_n(as.numeric(x))) +
  scale_fill_manual(values = c(
    "S3 dedup check"                       = COL_S3CHK,
    "Server execution (embedding + store)" = COL_SERVER
  ), name = NULL) +
  labs(
    title    = "End-to-end processing time — H100, ResNet-50",
    subtitle = "Stacked bars = client end-to-end (S3 dedup check + server execution)\nOrange diamonds: S3 upload time (reference, upstream of end-to-end)",
    x        = "Number of ingested objects",
    y        = "Time (s)"
  ) +
  theme_lat

p1
ggsave("log/latency_stacked.pdf", p1, width = 10, height = 6)

# ---------------------------------------------------------------------------
# Figure 2 – per-object throughput (ms/object) — log x
# ---------------------------------------------------------------------------

thr <- agg %>%
  select(n_processed, server_exec_ms_obj, end_to_end_ms_obj, upload_s3_ms_obj) %>%
  pivot_longer(-n_processed, names_to = "metric", values_to = "ms_per_obj") %>%
  mutate(metric = factor(metric,
    levels = c("upload_s3_ms_obj", "end_to_end_ms_obj", "server_exec_ms_obj"),
    labels = c("S3 upload", "Client end-to-end", "Server execution")))

p2 <- ggplot(thr, aes(x = n_processed, y = ms_per_obj,
                       colour = metric, linetype = metric)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_x_log10(
    breaks = unique(agg$n_processed),
    labels = fmt_n
  ) +
  scale_colour_manual(values = c(
    "S3 upload"        = COL_UPLOAD,
    "Client end-to-end"= "#4dac26",
    "Server execution" = COL_SERVER
  ), name = NULL) +
  scale_linetype_manual(values = c(
    "S3 upload"        = "dashed",
    "Client end-to-end"= "solid",
    "Server execution" = "solid"
  ), name = NULL) +
  labs(
    title    = "Per-object throughput — H100, ResNet-50",
    subtitle = "Fixed-cost amortisation as n grows (log x axis)",
    x        = "Number of objects (log scale)",
    y        = "Time per object (ms/object)"
  ) +
  theme_lat

ggsave("log/latency_per_object.pdf", p2, width = 10, height = 5)

# ---------------------------------------------------------------------------
# Figure 3 – throughput (objects/s) vs n_processed
# ---------------------------------------------------------------------------

thr2 <- agg %>%
  select(n_processed, throughput_server, throughput_e2e, throughput_upload) %>%
  pivot_longer(-n_processed, names_to = "metric", values_to = "obj_per_s") %>%
  mutate(metric = factor(metric,
    levels = c("throughput_upload", "throughput_e2e", "throughput_server"),
    labels = c("S3 upload", "Client end-to-end", "Server execution")))

p3 <- ggplot(thr2, aes(x = n_processed, y = obj_per_s,
                        colour = metric, linetype = metric)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_x_log10(breaks = unique(agg$n_processed), labels = fmt_n) +
  scale_colour_manual(values = c(
    "S3 upload"         = COL_UPLOAD,
    "Client end-to-end" = "#4dac26",
    "Server execution"  = COL_SERVER
  ), name = NULL) +
  scale_linetype_manual(values = c(
    "S3 upload"         = "dashed",
    "Client end-to-end" = "solid",
    "Server execution"  = "solid"
  ), name = NULL) +
  labs(
    title    = "Ingestion throughput — H100, ResNet-50",
    subtitle = "Objects per second; higher is better (log x axis)",
    x        = "Batch size (objects, log scale)",
    y        = "Throughput (objects / s)"
  ) +
  theme_lat

ggsave("log/latency_throughput.pdf", p3, width = 10, height = 5)

# ---------------------------------------------------------------------------
# Summary to console
# ---------------------------------------------------------------------------

cat("\n--- End-to-end decomposition (averages) ---\n")
agg %>%
  mutate(
    pct_server = round(server_exec / end_to_end * 100, 1),
    pct_s3chk  = round(s3_check    / end_to_end * 100, 1),
    pct_net    = round((end_to_end - server_exec - s3_check) / end_to_end * 100, 2)
  ) %>%
  select(n_processed, end_to_end, server_exec, pct_server, s3_check, pct_s3chk, pct_net, upload_s3) %>%
  print(n = Inf)

cat("\nPDFs saved in log/\n")

