#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --account=amc-general
#SBATCH --time=3-00:00:00
#SBATCH --output=output_for_nf_featurization-%j.out

module load nextflow

# featurization only
nextflow \
    featurization_only.nf \
    --fov_file "patient_well_fov.tsv" \
    --featurize_with_gpu false

# segmentation and featurization commented out for now
# nextflow \
#     pipeline_trials/featurization_only.nf \
#     --fov_file "patient_well_fov.tsv" \
#     --featurize_with_gpu false
