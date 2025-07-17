#!/bin/bash

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

rerun=$1


if [ "$rerun" == "rerun" ]; then
    txt_file="${git_root}/3.cellprofiling/load_data/rerun_combinations.txt"
else
    txt_file="${git_root}/3.cellprofiling/load_data/input_combinations.txt"
fi

# Check if TXT file exists
if [ ! -f "$txt_file" ]; then
    echo "Error: TXT file not found at $txt_file"
    exit 1
fi


# parse the txt_file where each line contains
# patient, well_fov, feature, compartment, channel, processor_type
while IFS= read -r line; do

    # split the line into an array
    IFS=$'\t' read -r -a parts <<< "$line"
    # assign the parts to variables
    patient="${parts[0]}"
    well_fov="${parts[1]}"
    feature="${parts[2]}"
    compartment="${parts[3]}"
    channel="${parts[4]}"
    processor_type="${parts[5]}"

    echo "Patient: $patient, WellFOV: $well_fov, Feature: $feature, Compartment: $compartment, Channel: $channel, UseGPU: $processor_type"


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
        --qos=normal \
        --account=amc-general \
        --time=5:00 \
        --output="logs/parents/featurize_parent_${patient}_${well_fov}_${feature}_${processor_type}_%j.out" \
        "$git_root"/3.cellprofiling/HPC_run_featurization_parent.sh \
        "$patient" \
        "$well_fov" \
        "$compartment" \
        "$channel" \
        "$feature" \
        "$processor_type"

done < "$txt_file"


echo "Featurization done"
