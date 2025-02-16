library(ggplot2)
library(dplyr)


snr_data_path <- file.path("../results/snr_data.csv")
snr_data <- read.csv(snr_data_path)
head(snr_data)

# min-max normalize the z.slice values for each well
snr_data <- snr_data %>%
  group_by(well) %>%
  mutate(z.slice = (z.slice - min(z.slice)) / (max(z.slice) - min(z.slice)))


# replace channel numbers with channel names
snr_data$channel <- gsub("405", "Hoechst", snr_data$channel)
snr_data$channel <- gsub("488", "ER", snr_data$channel)
snr_data$channel <- gsub("555", "AGP", snr_data$channel)
snr_data$channel <- gsub("640", "Mito", snr_data$channel)
snr_data$channel <- gsub("TRANS", "Brightfield", snr_data$channel)

width <- 10
height <- 5
options(repr.plot.width=width, repr.plot.height=height)
snr_plot <- (
    ggplot(snr_data, aes(x=z.slice, y=SNR, color=channel))
    + geom_line(aes(group_by=well))
    + labs(x="Z Slice", y="SNR", color="Channel")
    + theme_bw()
    + theme(
        legend.title=element_text(size=16),
        legend.text=element_text(size=14),
        axis.title=element_text(size=16),
        axis.text=element_text(size=14)
    )
    + facet_wrap(~channel)
)
print(snr_plot)
