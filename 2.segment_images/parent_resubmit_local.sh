#!/bin/bash
# activate segmentation environment

conda activate GFF_segmentation


# set arg to resubmit jobs or look for jobs to resubmit
# check if arg is passed
if [ "$#" -ne 0 ]; then
    if [ "$1" == "resubmit" ]; then
        resubmit="true"
        echo "Resubmitting jobs..."
        # remove the rerun file if it exists
        rerun_file="$(git rev-parse --show-toplevel)/2.segment_images/rerun_jobs.txt"
        if [ -f "$rerun_file" ]; then
            rm "$rerun_file"
        fi
    else
        echo "Invalid argument passed. Use 'resubmit' to resubmit jobs."
        echo "Use no arguments to check for jobs to resubmit."
        echo "Exiting..."
        exit 1
    fi
else
    resubmit="false"
    echo "No arguments passed. Checking for jobs to resubmit..."
    echo "But not resubmitting any jobs."
fi


git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

python "$git_root"/2.segment_images/scripts/10.perform_file_checks.py

# read the rerun file to get the patient and well_fov
rerun_file="$git_root"/2.segment_images/rerun_jobs.txt
if [ ! -f "$rerun_file" ]; then
    echo "Error: Rerun file $rerun_file does not exist."
    exit 1
fi
# extract the patient and well_fov from the rerun file
patient_well_fovs=()
while IFS=$'\t' read -r patient well_fov; do
    patient_well_fovs+=("$patient.$well_fov")
done < "$rerun_file"

# check the number of directories to process
total_dirs=${#patient_well_fovs[@]}
echo "Patient/Well_FOV pairs to process: ${patient_well_fovs[0]} to ${patient_well_fovs[-1]}"
echo "Total directories to process: $total_dirs"
# loop through all input directories
for patient_well_fov in "${patient_well_fovs[@]}"; do
    patient=$(echo "$patient_well_fov" | cut -d '.' -f 1)
    well_fov=$(echo "$patient_well_fov" | cut -d '.' -f 2)
    echo "Processing patient $patient, well_fov $well_fov"


    if [ "$resubmit" == "true" ]; then

        bash "${git_root}"/2.segment_images/child_segmentation.sh "$patient" "$well_fov"
    fi
done

# deactivate cellprofiler environment
conda deactivate

echo "All segmentation child jobs submitted"

