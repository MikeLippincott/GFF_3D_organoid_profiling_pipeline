list_of_packages <- c("ggplot2", "dplyr", "tidyr", "circlize")
for (package in list_of_packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(
                package,
                character.only = TRUE,
                quietly = TRUE,
                warn.conflicts = FALSE
            )
        )
    )
}

figures_path <- file.path("../figures/NF0014/")
if (!dir.exists(figures_path)) {
  dir.create(figures_path, recursive = TRUE)
}

umap_results <- arrow::read_parquet("../results/NF0014/3.organoid_fs_profiles_umap.parquet")
head(umap_results)

width <- 8
height <- 5
options(repr.plot.width = width, repr.plot.height = height)
umap_organoid_plot <- (
    ggplot(umap_results, aes(x = UMAP1, y = UMAP2, color = MOA))
    + geom_point(size = 3, alpha = 0.7)
    + scale_color_brewer(palette = "Paired")
    + labs(title = "UMAP of Organoid FS Profiles", x = "UMAP 0", y = "UMAP 1")
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.title = element_text(size = 14, hjust = 0.5),
        legend.text = element_text(size = 12)
    )
)
ggsave(umap_organoid_plot, file = "../figures/NF0014/umap_organoid_profiles.png", width = width, height = height, dpi = 300)
umap_organoid_plot


head(umap_results)

width <- 8
height <- 5
options(repr.plot.width = width, repr.plot.height = height)
umap_organoid_plot <- (
    ggplot(umap_results, aes(x = UMAP1, y = UMAP2, color = MOA, size = single_cell_count))
    + geom_point(alpha = 0.7)
    + scale_color_brewer(palette = "Paired")
    + labs(title = "UMAP of Organoid FS Profiles", x = "UMAP 0", y = "UMAP 1")
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.title = element_text(size = 14, hjust = 0.5),
        legend.text = element_text(size = 12)
    )
    + guides(
        size = guide_legend(
            title = "Single Cell Count",
            text = element_text(size = 16, hjust = 0.5)
            ),
        color = guide_legend(
            title = "MOA",
            text = element_text(size = 16, hjust = 0.5)
        )
    )
)
ggsave(umap_organoid_plot, file = "../figures/NF0014/umap_organoid_profiles_single_cell_count_per_organoid.png", width = width, height = height, dpi = 300)
umap_organoid_plot

umap_sc_results <- arrow::read_parquet('../results/NF0014/3.sc_fs_profiles_umap.parquet')
head(umap_sc_results)
umap_sc_plot <- (
    ggplot(umap_sc_results, aes(x = UMAP1, y = UMAP2, color = MOA))
    + geom_point(size = 3, alpha = 0.9)
    + scale_color_brewer(palette = "Paired")
    + labs(title = "UMAP of Single-Cell FS Profiles", x = "UMAP 0", y = "UMAP 1")
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.title = element_text(size = 14, hjust = 0.5),
        legend.text = element_text(size = 12)
    )
)
ggsave(umap_sc_plot, file = "../figures/NF0014/umap_sc_profiles.png", width = width, height = height, dpi = 300)

umap_sc_plot


# custom color palette - continuous
custom_palette <- colorRampPalette(c("blue", "green", "yellow"))
custom_colors <- custom_palette(200)
# make the scale continuous
custom_colors <- circlize::colorRamp2(seq(0, 1, length.out = 200), custom_colors)

umap_sc_plot <- (
    ggplot(umap_sc_results, aes(x = UMAP1, y = UMAP2, color = parent_organoid, shape = MOA))
    + geom_point(size = 3, alpha = 0.9)
    # add  custom color scale
    + scale_color_gradientn(colors = c("magenta", "green", "cyan", "orange", "blue"))
    + scale_shape_manual(values = c(
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
        ))  # different shapes for each MOA
    + labs(title = "UMAP of Single-Cell FS Profiles", x = "UMAP 0", y = "UMAP 1")
    + theme_bw()
    + theme(
        plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.title = element_text(size = 14, hjust = 0.5),
        legend.text = element_text(size = 12)
    )
    + guides(
        shape = guide_legend(
            title = "MOA",
            text = element_text(size = 16, hjust = 0.5)
            ),
        color = guide_legend(
            title = "Parent Organoid ID",
            text = element_text(size = 16, hjust = 0.5)
        )
    )
)
ggsave(umap_sc_plot, file = "../figures/NF0014/umap_sc_profiles_colored_by_parent_organoid.png", width = width, height = height, dpi = 300)

umap_sc_plot

