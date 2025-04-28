#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --account=amc-general
#SBATCH --time=7-00:00:00
#SBATCH --output=segmentation_parent-%j.out

# activate  cellprofiler environment
module load anaconda
conda init bash
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

patient=$1

echo "Processing patient $patient"

cd scripts/ || exit
# get all input directories in specified directory
z_stack_dir="../../data/$patient/zstack_images"
mapfile -t input_dirs < <(ls -d "$z_stack_dir"/*)
cd ../ || exit
total_dirs=$(echo "${input_dirs[@]}" | wc -w)
echo "Total directories: $total_dirs"
current_dir=0

touch segmentation.log
# loop through all input directories
for well_fov in "${input_dirs[@]}"; do
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    well_fov=$(basename "$dir")
    current_dir=$((current_dir + 1))
    echo -ne "Processing directory $current_dir of $total_dirs\r"
    echo "Beginning segmentation for $dir"
    sbatch child_segmentation.sh "$well_fov" "$patient"
done

# deactivate cellprofiler environment
conda deactivate

echo "All segmentation child jobs submitted"

