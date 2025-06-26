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

# output paths
organoid_features_path <- file.path(figures_path, "umap_organoid_features.png")
single_cell_features_path <- file.path(figures_path, "umap_single_cell_features.png")
single_cell_features_annotated_path <- file.path(figures_path, "umap_single_cell_w_parent_organoid_labels.png")

umap_results <- arrow::read_parquet("../results/NF0014/3.organoid_fs_profiles_umap.parquet")
head(umap_results)

# set custom colors for each MOA
custom_MOA_palette <- c(
    'Control' = "#5a5c5d",
    'Apoptosis' = "#882E8B",
    'MEK1/MEK2 Inhibitor'="#D700E0",
    'PI3K and HDAC inhibitor' = "#2E6B8B",
    'PI3K Inhibitor'="#0092E0",
    'Tyrosine kinase inhibtor'="#ECCC69",
    'Hsp90 Inhibition'="#E07E00",
    'mTOR inhibitor'="#ACE089",
    'IGF-1R inhibitor' = "#243036",
    'Na/K pump inhibition' = "#A16C28",
    'antihistamine' = "#3A8F00"
)


width <- 8
height <- 5
options(repr.plot.width = width, repr.plot.height = height)
umap_organoid_plot <- (
    ggplot(umap_results, aes(x = UMAP1, y = UMAP2, color = MOA, size = single_cell_count))
    + geom_point(alpha = 0.7)
    + scale_color_manual(values = custom_MOA_palette)
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
            text = element_text(size = 16, hjust = 0.5),
            override.aes = list(alpha = 1,size = 5)
        )
    )
)
ggsave(umap_organoid_plot, file = organoid_features_path, width = width, height = height, dpi = 300)
umap_organoid_plot

umap_sc_results <- arrow::read_parquet('../results/NF0014/3.sc_fs_profiles_umap.parquet')
head(umap_sc_results)
umap_sc_plot <- (
    ggplot(umap_sc_results, aes(x = UMAP1, y = UMAP2, color = MOA))
    + geom_point(size = 3, alpha = 0.9)
    + scale_color_manual(values = custom_MOA_palette)
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
        color = guide_legend(
            title = "MOA",
            text = element_text(size = 16, hjust = 0.5),
            override.aes = list(alpha = 1,size = 5)
        )
    )
)
ggsave(umap_sc_plot, file = single_cell_features_path, width = width, height = height, dpi = 300)

umap_sc_plot


umap_sc_results$unique_parent_organoid <- paste(
    umap_sc_results$image_set,
    umap_sc_results$parent_organoid,
    sep = "_"
)
# give a numerical label to each unique parent organoid
umap_sc_results$parent_organoid_label <- as.numeric(factor(umap_sc_results$unique_parent_organoid))

# custom color palette - continuous
custom_palette <- colorRampPalette(c("blue", "green", "yellow"))
custom_colors <- custom_palette(length(unique(umap_sc_results$parent_organoid_label)))
# make the scale continuous
custom_colors <- circlize::colorRamp2(seq(0, 1, length.out = length(unique(umap_sc_results$parent_organoid_label))), custom_colors)

umap_sc_plot <- (
    ggplot(umap_sc_results, aes(x = UMAP1, y = UMAP2, color = parent_organoid_label, shape = MOA))
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
ggsave(umap_sc_plot, file = single_cell_features_annotated_path, width = width, height = height, dpi = 300)

umap_sc_plot

