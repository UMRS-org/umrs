# Load necessary library
library(ggplot2)
library(scales)  # For percentage formatting
library(ggrepel)  # For better label placement

## Stat of my disk
#stage: BytesIO ; nb_objects: 10 ; Total_size: 47127.466MB ; total_time: 25216.854 ms ; Throughput: 1868.888 MB/s
#stage: BytesIO ; nb_objects: 320 ; Total_size: 1.971MB ; total_time: 214.083 ms ; Throughput: 9.208 MB/s

# Compute time per object for the large and small data
time_big <- 25216.854/10
time_small <- 214.083/320

# Compute size per object for the large and small data
size_big <- 47127.466/10
size_small <- 1.971/320

# Compute b (slope)
b <- (time_big - time_small) / (size_big - size_small)

# Compute a (intercept)
a <- time_big - b * size_big


# Compute bandwidth in MB/s (inverse of b in seconds)
bandwidth <- 1000 / b

# Print formatted output
cat(sprintf("Latency : %.3f ms and bandwidth : %.2f MB/s\n", a, bandwidth))

setwd("/Users/ejeannot/recherche/DDN_RIKEN/Unified_Multimodal_Retriever_Service/Infinia_demo/Perfomance_XP")

df <- read.csv("image_results.csv")
df_big <- read.csv("big_file_results.csv")

#find a and b such that the time to lad teh data is t = a + bS where s is the saise

# Compute time per object for the large and small data
time_big <- df_big$total_time_ms / df_big$nb_objects
time_small <- min(df$total_time_ms / df$nb_objects)

# Compute size per object for the large and small data
size_big <- df_big$Total_size_MB / df_big$nb_objects
size_small <- min(df$Total_size_MB / df$nb_objects)

# Compute b (slope)
b <- (time_big - time_small) / (size_big - size_small)

# Compute a (intercept)
a <- time_big - b * size_big


# Compute bandwidth in MB/s (inverse of b in seconds)
bandwidth <- 1000 / b

# Print formatted output
cat(sprintf("Latency : %.3f ms and bandwidth : %.2f MB/s\n", a, bandwidth))img_size_KB = min(df$Total_size_MB/df$nb_objects)*1024

# Compute time difference in milliseconds
time_Reading <- df$total_time_ms[df$stage == "BytesIO"] / min(df$nb_objects)
time_PIL <- (df$total_time_ms[df$stage == "BytesIO+PIL"] - df$total_time_ms[df$stage == "BytesIO"]) / min(df$nb_objects)
time_Embedding <- (df$total_time_ms[df$stage == "BytesIO+PIL+Embedding"] - df$total_time_ms[df$stage == "BytesIO+PIL"]) / min(df$nb_objects)
time_SQL_Store <- (df$total_time_ms[df$stage == "BytesIO+PIL+Embedding+SQL_Store"] - df$total_time_ms[df$stage == "BytesIO+PIL+Embedding"]) / min(df$nb_objects)


#time_Embedding <- 7

time_data <- data.frame(
  Steps = c("Reading From Infinia", "PIL Formating", "Embedding Computation", "SQL_Store"),
  Time = c(time_Reading, time_PIL, time_Embedding, time_SQL_Store)
)


# Ensure the order is preserved by setting factor levels
time_data$Steps <- factor(time_data$Steps, levels = rev(c("Reading From Infinia", "PIL Formating", "Embedding Computation", "SQL_Store")))

# Stacked Bar Plot
ggplot(time_data, aes(x="", y = Time, fill = Steps)) +
  geom_bar(stat = "identity") +
  labs(title = sprintf("Processing Time Breakdown for an image of %.2f KB", img_size_KB), x = "", y = "Time for one image (ms)") 

ggsave("proc_time_breakdown.pdf",height = 8, width = 6)

# Compute percentage
time_data$Percentage <- time_data$Time / sum(time_data$Time) * 100

ggplot(time_data, aes(x = "", y = Time, fill = Steps)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  labs(title = "Proportion of Time Spent in Each Steps") +
  geom_label_repel(aes(label = paste0(round(Percentage, 1), "%")),   # Ensure labels only inside the pie
                   position = position_stack(vjust = 0.5),
                   size = 5,
                   box.padding = 0.5,
                   point.padding = 0.5,
                   force = 5, 
                   direction = "y",
                   show.legend = FALSE)  # **Key Fix: Hides labels from the legend**

ggsave("proc_time_piechart.pdf",height = 5, width = 6)
