#!/bin/bash

module load anaconda
conda init bash
conda activate nf1_image_based_profiling_env

patient_array=( "NF0014" "NF0016" "NF0018" "NF0021" "SACRO219" )

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi
for patient in "${patient_array[@]}"; do

    python "$git_root"/4.processing_image_based_profiles/scripts/0.get_profiling_stats.py --patient "$patient"
    # get the list of all dirs in the parent_dir
    parent_dir="$git_root/data/$patient/extracted_features"
    # get the list of all dirs in the parent_dir
    dirs=$(ls -d "$parent_dir"/*)
    for dir in $dirs; do
        well_fov=$(basename "$dir")
        echo "$well_fov"
        python "$git_root"/4.processing_image_based_profiles/scripts/1.merge_feature_parquets.py --patient "$patient" --well_fov "$well_fov"
    done
done



conda deactivate

echo "All features merged for patients" "${patient_array[@]}"
