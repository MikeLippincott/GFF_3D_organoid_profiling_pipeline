#!/bin/bash

module load anaconda
conda init bash
conda activate nf1_image_based_profiling_env

patient=$1
well_fov=$2

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

python "$git_root"/4.processing_image_based_profiles/scripts/1.merge_feature_parquets.py --patient "$patient" --well_fov "$well_fov"


conda deactivate

echo "Patient $patient well_fov $well_fov completed"
