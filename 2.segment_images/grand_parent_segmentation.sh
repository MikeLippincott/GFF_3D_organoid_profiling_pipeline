#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --account=amc-general
#SBATCH --time=7-00:00:00
#SBATCH --output=segmentation_grandparent-%j.out

# activate  cellprofiler environment
module load anaconda
conda init bash
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

patient_array=( "NF0014" "NF0016" "NF0018" "NF0021" )

for patient in "${patient_array[@]}"; do
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done

    sbatch parent_segmentation.sh "$patient"
    sleep 100s # this allows for the parent and child jobs to submit prior to the next patient
done

conda deactivate

echo "All patients submitted for segmentation"
