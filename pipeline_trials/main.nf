#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.fov_file = 'patient_well_fov.tsv'

workflow {

    // Create the channel from the TSV file
    Channel
        .fromPath(params.fov_file)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            def patient = row.patient
            def well_fov = row.well_fov
            tuple(patient, well_fov)
        } \
        | segmentation
}

process segmentation {

    tag { "${patient}-${well_fov}" }

    input:
    tuple val(patient), val(well_fov)

    output:
    path "data/${patient}/processed_data/${well_fov}/cell_masks_decoupled.tiff"
    path "data/${patient}/processed_data/${well_fov}/cell_masks_reconstructed.tiff"
    path "data/${patient}/processed_data/${well_fov}/cell_reconstruction_dict.npy"
    path "data/${patient}/processed_data/${well_fov}/cytoplasm_mask.tiff"
    path "data/${patient}/processed_data/${well_fov}/nuclei_masks_decoupled.tiff"
    path "data/${patient}/processed_data/${well_fov}/nuclei_masks_reconstructed.tiff"
    path "data/${patient}/processed_data/${well_fov}/nuclei_reconstruction_dict.npy"
    path "data/${patient}/processed_data/${well_fov}/organoid_masks_decoupled.tiff"
    path "data/${patient}/processed_data/${well_fov}/organoid_masks_reconstructed.tiff"
    path "data/${patient}/processed_data/${well_fov}/organoid_reconstruction_dict.npy"

    script:
    """
    cd ../2.segment_images/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}"
    source child_segmentation_nf.sh $well_fov $patient
    cd ../pipeline_trials/ || exit 1
    """
}
