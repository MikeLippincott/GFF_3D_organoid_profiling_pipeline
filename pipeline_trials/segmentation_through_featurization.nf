#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.fov_file = 'patient_well_fov.tsv'
params.featurize_with_gpu = false

process segmentation {
    conda '/home/lippincm/miniforge3/envs/GFF_segmentation'

    tag { "${patient}-${well_fov}" }

    input:
    tuple val(patient), val(well_fov)

    output:
    tuple val(patient), val(well_fov), path("data/${patient}/processed_data/${well_fov}/")

    script:
    """
    cd ${baseDir}/../2.segment_images/ || exit 1
    echo "Processing patient: ${patient}, well_fov: ${well_fov}"
    source child_segmentation.sh ${well_fov} ${patient}
    cd ${baseDir}/ || exit 1
    """
}

process areasizeshape_cpu {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "areasizeshape_cpu" }

    input:
    tuple val(patient), val(well_fov), path(segmentation_output)

    output:
    path "areasizeshape_cpu.txt"

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1

    echo "Processing patient: ${patient}, well_fov: ${well_fov}, segmentation_output: ${segmentation_output}"
    source run_area_shape_child.sh ${patient} ${well_fov} FALSE ${segmentation_output}
    cd ${baseDir}/ || exit 1
    """
}

process areasizeshape_gpu {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "areasizeshape_gpu" }

    input:
    tuple val(patient), val(well_fov), path(segmentation_output)

    output:
    path "areasizeshape_gpu.txt"

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1

    echo "Processing patient: ${patient}, well_fov: ${well_fov}, segmentation_output: ${segmentation_output}"
    source run_area_shape_child.sh ${patient} ${well_fov} TRUE ${segmentation_output}
    cd ${baseDir}/ || exit 1
    """
}

// Define other featurizer processes similarly
process colocalization_cpu {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "colocalization_cpu" }

    input:
    tuple val(patient), val(well_fov), path(segmentation_output)

    output:
    path "colocalization_cpu.txt"

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1

    echo "Processing patient: ${patient}, well_fov: ${well_fov}, segmentation_output: ${segmentation_output}"
    source run_colocalization_child.sh ${patient} ${well_fov} FALSE ${segmentation_output}
    cd ${baseDir}/ || exit 1
    """
}

process colocalization_gpu {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "colocalization_gpu" }

    input:
    tuple val(patient), val(well_fov), path(segmentation_output)

    output:
    path "colocalization_gpu.txt"

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1

    echo "Processing patient: ${patient}, well_fov: ${well_fov}, segmentation_output: ${segmentation_output}"
    source run_colocalization_child.sh ${patient} ${well_fov} TRUE ${segmentation_output}
    cd ${baseDir}/ || exit 1
   """
}

process granularity_cpu {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "granularity_cpu" }

    input:
    tuple val(patient), val(well_fov), path(segmentation_output)

    output:
    path "granularity_cpu.txt"

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1

    echo "Processing patient: ${patient}, well_fov: ${well_fov}, segmentation_output: ${segmentation_output}"
    source run_granularity_child.sh ${patient} ${well_fov} FALSE ${segmentation_output}
    cd ${baseDir}/ || exit 1
    """
}

process granularity_gpu {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "granularity_gpu" }

    input:
    tuple val(patient), val(well_fov), path(segmentation_output)

    output:
    path "granularity_gpu.txt"

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1

    echo "Processing patient: ${patient}, well_fov: ${well_fov}, segmentation_output: ${segmentation_output}"
    source run_granularity_child.sh ${patient} ${well_fov} TRUE ${segmentation_output}
    cd ${baseDir}/ || exit 1
    """
}

process intensity_cpu {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "intensity_cpu" }

    input:
    tuple val(patient), val(well_fov), path(segmentation_output)

    output:
    path "intensity_cpu.txt"

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1

    echo "Processing patient: ${patient}, well_fov: ${well_fov}, segmentation_output: ${segmentation_output}"
    source run_intensity_child.sh ${patient} ${well_fov} FALSE ${segmentation_output}
    cd ${baseDir}/ || exit 1
    """
}

process intensity_gpu {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "intensity_gpu" }

    input:
    tuple val(patient), val(well_fov), path(segmentation_output)

    output:
    path "intensity_gpu.txt"

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1

    echo "Processing patient: ${patient}, well_fov: ${well_fov}, segmentation_output: ${segmentation_output}"
    source run_intensity_child.sh ${patient} ${well_fov} TRUE ${segmentation_output}
    cd ${baseDir}/ || exit 1
    """
}

process neighbors {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "neighbors" }

    input:
    tuple val(patient), val(well_fov), path(segmentation_output)

    output:
    path "neighbors.txt"

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1

    echo "Processing patient: ${patient}, well_fov: ${well_fov}, segmentation_output: ${segmentation_output}"
    source run_neighbors_child.sh ${patient} ${well_fov} ${segmentation_output}
    cd ${baseDir}/ || exit 1
    """
}

process texture {
    conda '/home/lippincm/miniforge3/envs/GFF_featurization'
    tag { "texture" }

    input:
    tuple val(patient), val(well_fov), path(segmentation_output)

    output:
    path "texture.txt"

    script:
    """
    cd ${baseDir}/../3.cellprofiling/slurm_scripts/ || exit 1

    echo "Processing patient: ${patient}, well_fov: ${well_fov}, segmentation_output: ${segmentation_output}"
    source run_texture_child.sh ${patient} ${well_fov} ${segmentation_output}
    cd ${baseDir}/ || exit 1
    """
}

// Add similar dependencies for other featurizer processes (colocalization, granularity, intensity, etc.)

workflow {
    // Common channel from input file
    def fov_ch = Channel
        .fromPath(params.fov_file)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            tuple(row.patient, row.well_fov)
        }

    // Run segmentation
    def segmentation_ch = fov_ch | segmentation

    // Attach featurize_with_gpu flag
    def full_ch = segmentation_ch.map { patient, well_fov, segmentation_output ->
        tuple(patient, well_fov, segmentation_output, params.featurize_with_gpu)
    }
    // get a persistent cpu_ch ofr the texture and neighbors
    def persistent_cpu_ch = full_ch.filter { patient, well_fov, segmentation_output, featurize_with_gpu -> !featurize_with_gpu }

    // Run CPU featurizers
    def cpu_ch = full_ch.filter { patient, well_fov, segmentation_output, featurize_with_gpu -> !featurize_with_gpu }
    cpu_ch | areasizeshape_cpu

    // Run GPU featurizers
    def gpu_ch = full_ch.filter { patient, well_fov, segmentation_output, featurize_with_gpu -> featurize_with_gpu }

    gpu_ch | areasizeshape_gpu
    gpu_ch | colocalization_gpu
    gpu_ch | granularity_gpu
    gpu_ch | intensity_gpu


    cpu_ch | colocalization_cpu
    cpu_ch | granularity_cpu
    cpu_ch | intensity_cpu

    persistent_cpu_ch | neighbors
    persistent_cpu_ch | texture

}
