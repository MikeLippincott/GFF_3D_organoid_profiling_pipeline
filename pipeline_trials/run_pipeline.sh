#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=output_for_nf_featurization-%j.out

module load nextflow

HPC_RUN=False
FEATS_ONLY=True


if [ "$HPC_RUN" = "True" ]; then
    if [ "$FEATS_ONLY" = "True" ]; then
        nextflow run \
            featurization_only.nf \
            --fov_file "patient_well_fov.tsv" \
            --featurize_with_gpu false \
            --conda_env_prefix "/projects/mlippincott@xsede.org/software/anaconda/envs/" \
            -c ./configs/nextflow.config \
            -profile SLURM_HPC
    else
        nextflow run \
            segmentation_through_featurization.nf \
            --fov_file "patient_well_fov.tsv" \
            --featurize_with_gpu false \
            --conda_env_prefix "/projects/mlippincott@xsede.org/software/anaconda/envs/" \
            -c ./configs/nextflow.config \
            -profile SLURM_HPC

    fi
else
    if [ "$FEATS_ONLY" = "True" ]; then
        nextflow run \
            featurization_only.nf \
            --fov_file "patient_well_fov.tsv" \
            --featurize_with_gpu false \
            -c ./configs/nextflow.config \
            --conda_env_prefix "/home/lippincm/miniforge3/envs/" \
            -profile local

    else
        nextflow run \
            segmentation_through_featurization.nf \
            --fov_file "patient_well_fov.tsv" \
            --featurize_with_gpu false \
            --conda_env_prefix "/home/lippincm/miniforge3/envs/" \
            -c ./configs/nextflow.config \
            -profile local
    fi
fi
