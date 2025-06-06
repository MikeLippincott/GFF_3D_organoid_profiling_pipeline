#!/usr/bin/env nextflow
// This is a nexflow script for running the GFF featurization pipeline
// It is designed to run on a SLURM cluster or locally
// Note that all code written here is written in nextflow DSL2
// DSL2 = Domain Specific Language 2
// For this pipeline to run it is assumed that segmentation has already been run

nextflow.enable.dsl = 2

params.fov_file = 'patient_well_fov.tsv'
params.featurize_with_gpu = false

include { SEGMENTATION } from '../modules/0.segmentation.nf'
include { SEGMENTATION_CLEANUP } from '../modules/1.segmentation_cleanup.nf'


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

    // Run segmentation
    segmented_ch | SEGMENTATION
    // Run segmentation cleanup
    segmented_ch | SEGMENTATION_CLEANUP
}
