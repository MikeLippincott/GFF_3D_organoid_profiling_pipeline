#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=2:00:00
#SBATCH --output=featurization_sc_grand_parent-%j.out


module load anaconda
conda init bash
conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

USE_GPU="FALSE"
patient=$1
patient="NF0014"

parent_dir="../../data/$patient/cellprofiler"
# get the list of all dirs in the parent_dir
dirs=$(ls -d $parent_dir/*)

cd ../ || exit

# loop through each dir and submit a job
for dir in $dirs; do
    well_fov=$(basename $dir)
    echo $well_fov
    # check that the number of jobs is less than 990
    # prior to submitting a job
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    sbatch HPC_run_featurization_parent.sh "$well_fov" $USE_GPU $patient

done

conda deactivate

echo "Featurization done"
