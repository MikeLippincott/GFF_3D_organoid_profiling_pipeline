#!/bin/bash

module load anaconda
conda init bash
conda activate nf1_image_based_profiling_env

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

patient_array_file_path="$git_root/data/patient_IDs.txt"
# read the patient IDs from the file into an array
if [[ -f "$patient_array_file_path" ]]; then
    readarray -t patient_array < "$patient_array_file_path"
else
    echo "Error: File $patient_array_file_path does not exist."
    exit 1
fi

for patient in "${patient_array[@]}"; do
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
    --time=1:00:00 \
    --output=featurization_sc_grand_parent-%j.out \
    "$git_root"/4.processing_image_based_profiles/HPC_run_featurization_grand_parent.sh "$patient"
done

conda deactivate

echo "All patients submitted for segmentation"
