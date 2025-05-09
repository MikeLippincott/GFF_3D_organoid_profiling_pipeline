#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=2:00:00
#SBATCH --output=featurization_sc_great_grand_parent-%j.out


module load anaconda
conda init bash
conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

patient_array=( "NF0014" "NF0016" "NF0018" "NF0021" "SACRO219" )

for patient in "${patient_array[@]}"; do
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done

    sbatch \
    --nodes=1 \
    --ntasks=1 \
    --partition=amilan \
    --qos=normal \
    --account=amc-general \
    --time=1:00:00 \
    --output=featurization_sc_grand_parent-%j.out \
    HPC_run_featurization_grand_parent.sh "$patient"
done

conda deactivate

echo "All patients submitted for segmentation"

