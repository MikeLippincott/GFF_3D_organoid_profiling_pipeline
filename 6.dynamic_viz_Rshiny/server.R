# Load necessary libraries
library(shiny)
library(ggplot2)
library(dplyr)
library(arrow)
library(rlang)


# Define server logic
server <- function(input, output, session) {
    # UI elements
    output$sc_or_organoid <- renderUI({
        selectInput("sc_or_organoid", "Select Data Type:",
                   choices = c("Single Cell", "Organoid"),
                   selected = "Organoid")
    })

    output$data_level <- renderUI({
        selectInput("data_level", "Select Data Level:",
                   choices = c("Consensus", "Feature Selected"),
                   selected = "Feature Selected")
    })

    # Reactive data loading
    df <- reactive({
        req(input$sc_or_organoid, input$data_level)

        tryCatch({
            if (input$sc_or_organoid == "Single Cell") {
                # Load single cell data
                if (input$data_level == "Consensus") {
                    data <- arrow::read_parquet("data/sc_consensus_umap.parquet")
                } else if (input$data_level == "Feature Selected") {
                    data <- arrow::read_parquet("data/sc_fs_umap.parquet")
                }
            } else if (input$sc_or_organoid == "Organoid") {
                # Load organoid data
                if (input$data_level == "Consensus") {
                    data <- arrow::read_parquet("data/organoid_consensus_umap.parquet")
                } else if (input$data_level == "Feature Selected") {
                    data <- arrow::read_parquet("data/organoid_fs_umap.parquet")
                }
            }

            # Debug: print column names
            cat("Column names:", paste(colnames(data), collapse = ", "), "\n")

            return(data)
        }, error = function(e) {
            cat("Error loading data:", e$message, "\n")
            return(NULL)
        })
    })

    output$PatientSelect <- renderUI({
        req(df())

        # Try different possible patient column names
        patient_col <- if("patient" %in% colnames(df())) {
            "patient"
        } else if("Patient" %in% colnames(df())) {
            "Patient"
        } else if("Metadata_Patient" %in% colnames(df())) {
            "Metadata_Patient"
        } else {
            NULL
        }

        if (!is.null(patient_col)) {
            patients <- unique(df()[[patient_col]])
            patients <- patients[!is.na(patients)]  # Remove NA values
            patients <- sort(patients)  # Sort alphabetically

            tagList(
                checkboxGroupInput("PatientSelect", "Select Patient(s):",
                                  choices = patients,
                                  selected = patients),
                fluidRow(
                    column(6, actionButton("selectAllPatients", "Select All",
                                         style = "width: 100%; font-size: 12px;")),
                    column(6, actionButton("clearAllPatients", "Clear All",
                                         style = "width: 100%; font-size: 12px;"))
                )
            )
        } else {
            h4("No patient column found in data")
        }
    })

    output$FacetSelect <- renderUI({
        # give the choice to facet by patient or not
        selectInput("FacetSelect", "Facet by Patient:", choices = c("Yes", "No"), selected = "No")
    })

    output$TreatmentSelect <- renderUI({
        req(df())

        # Try different possible treatment column names
        treatment_col <- if("treatment" %in% colnames(df())) {
            "treatment"
        } else if("Treatment" %in% colnames(df())) {
            "Treatment"
        } else if("Target" %in% colnames(df())) {
            "Target"
        } else if("target" %in% colnames(df())) {
            "target"
        } else {
            NULL
        }

        if (!is.null(treatment_col)) {
            treatments <- unique(df()[[treatment_col]])
            treatments <- treatments[!is.na(treatments)]  # Remove NA values
            treatments <- sort(treatments)  # Sort alphabetically

            tagList(
                checkboxGroupInput("TreatmentSelect", "Select Treatment(s):",
                                  choices = treatments,
                                  selected = treatments),
                fluidRow(
                    column(6, actionButton("selectAllTreatments", "Select All",
                                         style = "width: 100%; font-size: 12px;")),
                    column(6, actionButton("clearAllTreatments", "Clear All",
                                         style = "width: 100%; font-size: 12px;"))
                )
            )
        } else {
            h4("No treatment column found in data")
        }
    })

    # Observers for select all / clear all patients
    observeEvent(input$selectAllPatients, {
        req(df())
        patient_col <- if("patient" %in% colnames(df())) {
            "patient"
        } else if("Patient" %in% colnames(df())) {
            "Patient"
        } else if("Metadata_Patient" %in% colnames(df())) {
            "Metadata_Patient"
        } else {
            NULL
        }

        if (!is.null(patient_col)) {
            patients <- unique(df()[[patient_col]])
            patients <- patients[!is.na(patients)]
            updateCheckboxGroupInput(session, "PatientSelect", selected = patients)
        }
    })

    observeEvent(input$clearAllPatients, {
        updateCheckboxGroupInput(session, "PatientSelect", selected = character(0))
    })

    # Observers for select all / clear all treatments
    observeEvent(input$selectAllTreatments, {
        req(df())
        treatment_col <- if("treatment" %in% colnames(df())) {
            "treatment"
        } else if("Treatment" %in% colnames(df())) {
            "Treatment"
        } else if("Target" %in% colnames(df())) {
            "Target"
        } else if("target" %in% colnames(df())) {
            "target"
        } else {
            NULL
        }

        if (!is.null(treatment_col)) {
            treatments <- unique(df()[[treatment_col]])
            treatments <- treatments[!is.na(treatments)]
            updateCheckboxGroupInput(session, "TreatmentSelect", selected = treatments)
        }
    })

    observeEvent(input$clearAllTreatments, {
        updateCheckboxGroupInput(session, "TreatmentSelect", selected = character(0))
    })

    # Filtered data based on patient and treatment selection
    filtered_data <- reactive({
        req(df(), input$PatientSelect)

        data <- df()

        # Find patient column
        patient_col <- if("patient" %in% colnames(data)) {
            "patient"
        } else if("Patient" %in% colnames(data)) {
            "Patient"
        } else if("Metadata_Patient" %in% colnames(data)) {
            "Metadata_Patient"
        } else {
            NULL
        }

        # Find treatment column
        treatment_col <- if("treatment" %in% colnames(data)) {
            "treatment"
        } else if("Treatment" %in% colnames(data)) {
            "Treatment"
        } else if("Target" %in% colnames(data)) {
            "Target"
        } else if("target" %in% colnames(data)) {
            "target"
        } else {
            NULL
        }

        # Filter by patient
        if (!is.null(patient_col)) {
            data <- dplyr::filter(data, !!sym(patient_col) %in% input$PatientSelect)
        }

        # Filter by treatment if selection exists
        if (!is.null(treatment_col) && !is.null(input$TreatmentSelect)) {
            data <- dplyr::filter(data, !!sym(treatment_col) %in% input$TreatmentSelect)
        }

        return(data)
    })

    output$umapPlot <- renderPlot({
        req(filtered_data())

        # set custom colors for each MOA
        custom_MOA_palette <- c(
            'Control' = "#5a5c5d",
            'MEK1/2 inhibitor' = "#882E8B",


            'HDAC inhibitor' = "#1E6B61",
            'PI3K and HDAC inhibitor' = "#2E6B8B",
            'PI3K inhibitor'="#0092E0",

            'receptor tyrosine kinase inhibitor' = "#576A20",
            'tyrosine kinase inhibitor' = "#646722",

            'mTOR inhibitor' = "#ACE089",
            'IGF-1R inhibitor' = "#ACE040",

            'HSP90 inhibitor'="#33206A",
            'Apoptosis induction'="#272267",
            'Na+/K+ pump inhibitor' = "#A16C28",
            'histamine H1 receptor antagonist' = "#3A8F00",
            'DNA binding' = "#174F17",
            'BRD4 inhibitor' = "#ff0000"

        )


        data <- filtered_data()

        # Check for UMAP columns with different possible names
        umap1_col <- if("UMAP1" %in% colnames(data)) {
            "UMAP1"
        } else if("UMAP_1" %in% colnames(data)) {
            "UMAP_1"
        } else if("umap1" %in% colnames(data)) {
            "umap1"
        } else {
            NULL
        }

        umap2_col <- if("UMAP2" %in% colnames(data)) {
            "UMAP2"
        } else if("UMAP_2" %in% colnames(data)) {
            "UMAP_2"
        } else if("umap2" %in% colnames(data)) {
            "umap2"
        } else {
            NULL
        }

        target_col <- if("Target" %in% colnames(data)) {
            "Target"
        } else if("target" %in% colnames(data)) {
            "target"
        } else if("treatment" %in% colnames(data)) {
            "treatment"
        } else {
            NULL
        }

        # Find patient column for coloring/faceting
        patient_col <- if("patient" %in% colnames(data)) {
            "patient"
        } else if("Patient" %in% colnames(data)) {
            "Patient"
        } else if("Metadata_Patient" %in% colnames(data)) {
            "Metadata_Patient"
        } else {
            NULL
        }

        if (is.null(umap1_col) || is.null(umap2_col)) {
            # Create a simple plot showing available columns
            plot(1, 1, type = "n", xlab = "", ylab = "")
            text(1, 1, "Check column names in data")
            return()
        }

        req(input$FacetSelect)  # Ensure FacetSelect input exists

        # Debug: print what we found
        cat("UMAP1 column:", umap1_col, "\n")
        cat("UMAP2 column:", umap2_col, "\n")
        cat("Target column:", target_col, "\n")
        cat("Patient column:", patient_col, "\n")
        cat("FacetSelect value:", input$FacetSelect, "\n")
        cat("Selected patients:", paste(input$PatientSelect, collapse = ", "), "\n")
        cat("Selected treatments:", paste(input$TreatmentSelect, collapse = ", "), "\n")
        cat("Patient column class:", class(data[[patient_col]]), "\n")
        cat("Unique patients:", paste(unique(data[[patient_col]]), collapse = ", "), "\n")
        cat("Number of rows in data:", nrow(data), "\n")

        # Check if all required columns exist
        if (is.null(target_col)) {
            plot(1, 1, type = "n", xlab = "", ylab = "")
            text(1, 1, "Target column not found")
            return()
        }

        if (input$FacetSelect == "No") {
            # Create plot with dynamic column names
            p <- (ggplot(data, aes(x = !!sym(umap1_col), y = !!sym(umap2_col),
                                color = !!sym(target_col))) +
                geom_point(size = 1, alpha = 0.7) +
                theme_minimal() +
                labs(x = umap1_col, y = umap2_col, title = "UMAP Plot") +
                scale_color_manual(values = custom_MOA_palette) + # Use custom colors
                theme_minimal() +
                        labs(x = umap1_col, y = umap2_col, title = "UMAP Plot") +
                        scale_color_manual(values = custom_MOA_palette) # Use custom colors
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
                        ))
        } else {
            # Create plot with dynamic column names and faceting
            if (!is.null(patient_col)) {
                # Ensure patient column is a factor and clean any problematic values
                data[[patient_col]] <- as.factor(as.character(data[[patient_col]]))

                # Remove any rows with NA in patient column
                data <- data[!is.na(data[[patient_col]]), ]

                if (nrow(data) == 0) {
                    plot(1, 1, type = "n", xlab = "", ylab = "")
                    text(1, 1, "No data after filtering")
                    return()
                }

                cat("After cleaning - rows:", nrow(data), "patients:", paste(unique(data[[patient_col]]), collapse = ", "), "\n")

                # Try a simpler approach to faceting
                tryCatch({
                    p <- (ggplot(data, aes(x = !!sym(umap1_col), y = !!sym(umap2_col),
                                        color = !!sym(target_col))) +
                        geom_point(size = 1, alpha = 0.7) +
                        theme_minimal() +
                        labs(x = umap1_col, y = umap2_col, title = "UMAP Plot") +
                        scale_color_manual(values = custom_MOA_palette) # Use custom colors
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

                    # Add faceting using a different method
                    if (patient_col == "patient") {
                        p <- p + facet_wrap(~ patient, scales = "free")
                    } else {
                        # Fallback to no faceting if column name is unusual
                        cat("Using alternative faceting approach\n")
                        p <- p + labs(title = paste("UMAP Plot - Multiple Patients:", paste(unique(data[[patient_col]]), collapse = ", ")))
                    }
                }, error = function(e) {
                    cat("Faceting error:", e$message, "\n")
                    # Fallback: create plot without faceting
                    p <- (ggplot(data, aes(x = !!sym(umap1_col), y = !!sym(umap2_col),
                                        color = !!sym(target_col))) +
                        geom_point(size = 1, alpha = 0.7) +
                        theme_bw() +
                        labs(x = umap1_col, y = umap2_col, title = "UMAP Plot (Faceting Failed)") +
                        scale_color_manual(values = custom_MOA_palette)
                        + geom_point(alpha = 0.7, size = 1)
                        # make the colors continuous
                        + scale_color_gradient(low = "lightblue", high = "darkblue")
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
                })
            } else {
                # Fallback when no patient column - just show regular plot
                p <- (ggplot(data, aes(x = !!sym(umap1_col), y = !!sym(umap2_col),
                                    color = !!sym(target_col))) +
                    geom_point(size = 1, alpha = 0.7) +
                    theme_bw() +
                    labs(x = umap1_col, y = umap2_col, title = "UMAP Plot (No Patient Column Found)") +
                    scale_color_manual(values = custom_MOA_palette)
                    + geom_point(alpha = 0.7)
                    # make the colors continuous
                    + scale_color_gradient(low = "lightblue", high = "darkblue")
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
            }
        }

        p
    })
}
