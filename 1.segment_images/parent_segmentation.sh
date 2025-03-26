#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=24:00:00
#SBATCH --output=segmentation-%j.out

# activate  cellprofiler environment
module load miniforge
conda init bash
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


cd scripts/ || exit
# get all input directories in specified directory
z_stack_dir="../../data/NF0014/zstack_images"
# z_stack_dir="../../data/NF0014/test_dir/"
mapfile -t input_dirs < <(ls -d "$z_stack_dir"/*)
cd ../ || exit
total_dirs=$(echo "${input_dirs[@]}" | wc -w)
echo "Total directories: $total_dirs"
current_dir=0

touch segmentation.log
# loop through all well_fov directories and submit a job for each
for dir in "${input_dirs[@]}"; do
    dir=${dir%*/}
    well_fov=$(basename "$dir")
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    current_dir=$((current_dir + 1))
    echo -ne "Processing directory $current_dir of $total_dirs\r"
    echo "Beginning segmentation for $well_fov"
    sbatch child_segmentation.sh "$well_fov"
done

echo "Cleaning up segmentation files"
python 7.clean_up_segmentation.py >> segmentation.log
echo -ne "\n"

# deactivate cellprofiler environment
conda deactivate

echo "Segmentation complete"
