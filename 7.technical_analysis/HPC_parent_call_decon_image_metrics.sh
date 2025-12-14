#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --account=amc-general
#SBATCH --time=7-00:00:00
#SBATCH --output=decon_image_metrics_grandparent-%j.out

load_file_path="./loadfiles/decon_image_metrics_load_file.txt"
while IFS= read -r line; do
    # split the line into an array
    IFS=$'\t' read -r -a parts <<< "$line"
    # assign the parts to variables
    patient="${parts[0]}"
    well_fov="${parts[1]}"
    echo "Submitting decon image metrics for Patient: $patient, WellFOV: $well_fov"

    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 5s
    done

    sbatch \
        --nodes=1 \
        --ntasks=3 \
        --partition=aa100 \
        --gres=gpu:1 \
        --qos=normal \
        --account=amc-general \
        --time=1:30:00 \
        --export=patient="$patient",well_fov="$well_fov" \
        --output=logs/decon_image_metrics_"$patient"_"$well_fov"_%j.out \
        HPC_child_call_decon_image_metrics.sh "$patient" "$well_fov"

done < "$load_file_path"

echo "All segmentation child jobs submitted"

