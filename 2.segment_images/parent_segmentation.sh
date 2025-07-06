#!/bin/bash
# activate segmentation environment
module load anaconda
conda init bash
conda activate GFF_segmentation

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

patient=$1

echo "Processing patient $patient"
if [ -z "$patient" ]; then
    echo "Error: Patient argument is required."
    exit 1
fi

# get all input directories in specified directory
z_stack_dir="$git_root/data/$patient/zstack_images"
mapfile -t input_dirs < <(ls -d "$z_stack_dir"/*)
total_dirs=$(echo "${input_dirs[@]}" | wc -w)
echo "Total directories: $total_dirs"
current_dir=0

# loop through all input directories
for well_fov in "${input_dirs[@]}"; do
    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done
    well_fov=$(basename "$well_fov")
    current_dir=$((current_dir + 1))
    echo -ne "Processing directory $current_dir of $total_dirs\r"
    echo "Beginning segmentation for $well_fov"
    sbatch \
        --nodes=1 \
        --ntasks=3 \
        --partition=aa100 \
        --gres=gpu:1 \
        --qos=normal \
        --account=amc-general \
        --time=20:00 \
        --output=segmentation_child-%j.out \
        "${git_root}"/2.segment_images/child_segmentation.sh "$patient" "$well_fov"

done

# deactivate cellprofiler environment
conda deactivate

echo "All segmentation child jobs submitted"

