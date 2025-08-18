#!/bin/bash

conda activate gff_preprocessing_env
git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


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

    source \
        "$git_root"/3.cellprofiling/local_run_featurization_parent.sh \
        "$patient" \
        "$well_fov" \
        "$compartment" \
        "$channel" \
        "$feature" \
        "$processor_type"

done < "$txt_file"


echo "Featurization done"
