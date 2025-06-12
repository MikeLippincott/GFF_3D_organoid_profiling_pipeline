#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=output_for_nf_featurization-%j.out

HPC_RUN=False
FEATS_ONLY=True

patient="$1"

load_file="./load_files/${patient}_well_fov.tsv"


if [ "$HPC_RUN" = "True" ]; then
    module load nextflow
    if [ "$FEATS_ONLY" = "True" ]; then
        nextflow run \
            workflows/featurization_only.nf \
            --fov_file "${load_file}" \
            --featurize_with_gpu false \
            -c ./configs/nextflow.config \
            -profile SLURM_HPC
    else
        nextflow run \
            workflows/segmentation_through_featurization.nf \
            --fov_file "${load_file}" \
            --featurize_with_gpu false \
            -c ./configs/nextflow.config \
            -profile SLURM_HPC

    fi
else
    if [ "$FEATS_ONLY" = "True" ]; then
        nextflow run \
            workflows/featurization_only.nf \
            --fov_file "${load_file}" \
            --featurize_with_gpu false \
            -c ./configs/nextflow.config \
            -profile local

    else
        nextflow run \
            workflows/segmentation_through_featurization.nf \
            --fov_file "${load_file}" \
            --featurize_with_gpu false \
            -c ./configs/nextflow.config \
            -profile local
    fi
fi
