#!/bin/bash

USE_GPU="FALSE"
patient=$1

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

json_file="${git_root}/3.cellprofiling/load_data/input_combinations.json"

# Check if JSON file exists
if [ ! -f "$json_file" ]; then
    echo "Error: JSON file not found at $json_file"
    exit 1
fi

parent_dir="${git_root}/data/$patient/zstack_images"
# get the list of all dirs in the parent_dir
dirs=$(ls -d "$parent_dir"/*)

jq -r '.[] | "\(.feature) \(.compartment) \(.channel)"' "$json_file" | while read -r feature compartment channel; do

    # loop through each dir and submit a job
    for dir in $dirs; do
        well_fov=$(basename "$dir")
        echo "$well_fov"
        # check that the number of jobs is less than 990
        # prior to submitting a job
        number_of_jobs=$(squeue -u "$USER" | wc -l)
        while [ "$number_of_jobs" -gt 990 ]; do
            sleep 1s
            number_of_jobs=$(squeue -u "$USER" | wc -l)
        done
        sbatch \
            --nodes=1 \
            --ntasks=1 \
            --partition=amilan \
            --qos=long \
            --account=amc-general \
            --time=7-00:00:00 \
            --output=parent_featurize-%j.out \
            "$git_root"/3.cellprofiling/HPC_run_featurization_parent.sh \
            "$patient" \
            "$well_fov" \
            "$compartment" \
            "$channel" \
            "$feature" \
            "$USE_GPU"
    done
done


echo "Featurization done"
