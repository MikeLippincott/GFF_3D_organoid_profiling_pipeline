#!/bin/bash

module load anaconda
conda init bash
conda activate nf1_image_based_profiling_env

patient=$1
well_fov=$2

echo "Patient: $patient, Well/FOV: $well_fov"

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

log_file="$git_root/4.processing_image_based_profiles/logs/${patient}_${well_fov}.log"
touch "$log_file"  # create the log file if it doesn't exist
{
    python "$git_root"/4.processing_image_based_profiles/scripts/1.merge_feature_parquets.py --patient "$patient" --well_fov "$well_fov"
    python "$git_root"/4.processing_image_based_profiles/scripts/2.merge_sc.py --patient "$patient" --well_fov "$well_fov"
    python "$git_root"/4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py --patient "$patient" --well_fov "$well_fov"
} >> "$log_file" 2>&1

conda deactivate

echo "Patient $patient well_fov $well_fov completed"
