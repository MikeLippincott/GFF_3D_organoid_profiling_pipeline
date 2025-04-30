#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.fov_file = 'patient_well_fov.tsv'
params.featurize_with_gpu = false


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
    source child_segmentation.sh ${well_fov} ${patient}
    cd ../../pipeline_trials/ || exit 1
    """
}

process areasizeshape_cpu {

    tag { "areasizeshape_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout into dummy_output_ch

    script:
    """
    cd ../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}, use_gpu: ${featurize_with_gpu}"
    source run_area_shape_child.sh ${patient} ${well_fov} FALSE ${featurize_with_gpu}
    cd ../../pipeline_trials/ || exit 1
    """
}

process areasizeshape_gpu {

    tag { "areasizeshape_gpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout into dummy_output_ch

    script:
    """
    cd ../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}, use_gpu: ${featurize_with_gpu}"
    source run_area_shape_child.sh ${patient} ${well_fov} TRUE ${featurize_with_gpu}
    cd ../../pipeline_trials/ || exit 1
    """
}

process colocalization_cpu {
    tag { "colocalization_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout into dummy_output_ch

    script:
    """
    cd ../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}"
    source run_colocalization_child.sh ${patient} FALSE ${well_fov}
    cd ../../pipeline_trials/ || exit 1
    """
}

process colocalization_gpu {
    tag { "colocalization_gpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout into dummy_output_ch

    script:
    """
    cd ../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}"
    source run_colocalization_child.sh ${patient} TRUE ${well_fov}
    cd ../../pipeline_trials/ || exit 1
    """
}

process granularity_cpu {
    tag { "granularity_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout into dummy_output_ch

    script:
    """
    cd ../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}"
    source run_granularity_child.sh ${patient} FALSE ${well_fov}
    cd ../../pipeline_trials/ || exit 1
    """
}

process granularity_gpu {
    tag { "granularity_gpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout into dummy_output_ch

    script:
    """
    cd ../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}"
    source run_granularity_child.sh ${patient} TRUE ${well_fov}
    cd ../../pipeline_trials/ || exit 1
    """
}

process intensity_cpu {
    tag { "intensity_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout into dummy_output_ch

    script:
    """
    cd ../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Running GPU featurization for patient: ${patient}, well_fov: ${well_fov} use_gpu: ${featurize_with_gpu}"
    source run_intensity_child.sh ${well_fov} FALSE ${patient}
    cd ../../pipeline_trials/ || exit 1
    """
}

process intensity_gpu {
    tag { "intensity_gpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout into dummy_output_ch

    script:
    """
    cd ../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Running GPU featurization for patient: ${patient}, well_fov: ${well_fov} use_gpu: ${featurize_with_gpu}"
    source run_intensity_child.sh ${well_fov} TRUE ${patient}
    cd ../../pipeline_trials/ || exit 1
    """
}



process neighbors_cpu {
    tag { "neighbors_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout into dummy_output_ch

    script:
    """
    cd ../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Running GPU featurization for patient: ${patient}, well_fov: ${well_fov} use_gpu: ${featurize_with_gpu}"
    source run_neighbors_child.sh ${well_fov} FALSE ${patient}
    cd ../../pipeline_trials/ || exit 1
    """
}

process texture_cpu {
    tag { "texture_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout into dummy_output_ch

    script:
    """
    cd ../3.cellprofiling/slurm_scripts/ || exit 1
    echo "Running CPU featurization for patient: ${patient}, well_fov: ${well_fov} use_gpu: ${featurize_with_gpu}"
    source run_texture_child.sh ${well_fov} FALSE ${patient}
    cd ../../pipeline_trials/ || exit 1
    """
}

workflow {

    // Common channel from input file
    def fov_ch = Channel
        .fromPath(params.fov_file)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            tuple(row.patient, row.well_fov, params.featurize_with_gpu)
        }

    // Run segmentation (shared between all)
    def segmented_ch = fov_ch.map { patient, well_fov, _ -> tuple(patient, well_fov) } | segmentation

    // Re-attach featurize_with_gpu flag
    def full_ch = segmented_ch.map { patient, well_fov -> tuple(patient, well_fov, params.featurize_with_gpu) }

    // always run CPU branches
    def persistent_ch = full_ch

    // Split full channel into two: GPU and CPU
    def (gpu_ch, cpu_ch) = full_ch.split { it[2] }

    // Run GPU branches
    gpu_ch | areasizeshape_gpu
    gpu_ch | colocalization_gpu
    gpu_ch | granularity_gpu
    gpu_ch | intensity_gpu

    // Run CPU branches
    cpu_ch | areasizeshape_cpu
    cpu_ch | colocalization_cpu
    cpu_ch | granularity_cpu
    cpu_ch | intensity_cpu

    // Run neighbors and texture on CPU
    persistent_ch | neighbors_cpu
    persistent_ch | texture_cpu
}
