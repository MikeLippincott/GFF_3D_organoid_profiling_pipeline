#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.fov_file = 'patient_well_fov.tsv'
params.featurize_with_gpu = false

process areasizeshape_cpu {

    tag { "areasizeshape_cpu" }

    input:
    tuple val(patient), val(well_fov), val(featurize_with_gpu)

    output:
    stdout emit: dummy_output_ch_txt

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
    stdout emit: dummy_output_ch_txt

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
    stdout emit: dummy_output_ch_txt

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
    stdout emit: dummy_output_ch_txt

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
    stdout emit: dummy_output_ch_txt

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
    stdout emit: dummy_output_ch_txt

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
    stdout emit: dummy_output_ch_txt

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
    stdout emit: dummy_output_ch_txt

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
    stdout emit: dummy_output_ch_txt

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
    stdout emit: dummy_output_ch_txt

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
            tuple(row.patient, row.well_fov, params.featurize_with_gpu ?: false) // Ensure featurize_with_gpu is not null
        }

    // Run segmentation (shared between all)
    def segmented_ch = fov_ch.map { patient, well_fov, _ -> tuple(patient, well_fov) }

    // Re-attach featurize_with_gpu flag
    def full_ch = segmented_ch.map { patient, well_fov ->
        tuple(patient, well_fov, params.featurize_with_gpu)
    }

    // if featurize_with_gpu is false, run CPU branches
    def cpu_ch = full_ch.filter { patient, well_fov, featurize_with_gpu -> !featurize_with_gpu }
    // if featurize_with_gpu is true, run GPU branches
    def gpu_ch = full_ch.filter { patient, well_fov, featurize_with_gpu -> featurize_with_gpu }
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

    // Always run texture on CPU
    segmented_ch.map { patient, well_fov -> tuple(patient, well_fov, false) } | texture_cpu
}
