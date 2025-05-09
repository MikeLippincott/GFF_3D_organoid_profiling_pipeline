#!/bin/bash


module load anaconda
conda init bash
conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


USE_GPU="FALSE"
patient=$1

parent_dir="../data/$patient/cellprofiler"
# get the list of all dirs in the parent_dir
dirs=$(ls -d $parent_dir/*)


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
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=long \
        --account=amc-general \
        --time=7-00:00:00 \
        --output=parent_featurize-%j.out \
        HPC_run_featurization_parent.sh "$well_fov" $USE_GPU $patient

done

conda deactivate

echo "Featurization done"
