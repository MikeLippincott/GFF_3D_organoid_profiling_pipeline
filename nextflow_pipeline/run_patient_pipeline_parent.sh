#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=1:00:00
#SBATCH --output=output_for_nf_featurization-%j.out

HPC_RUN=True
FEATS_ONLY=True

module load anaconda
conda init bash
conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


patient_array=( "NF0014" "NF0016" "NF0018" "NF0021" "SARCO219" "SARCO361" )

for patient in "${patient_array[@]}"; do
    cd scripts || exit 1
    python 0.specify_patient_well_fov.py --patient "$patient"
    cd .. || exit 1
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=long \
        --account=amc-general \
        --time=7-00:00:00 \
        --output=parent_featurize-%j.out \
        nextflow_pipeline/run_single_patient_pipeline.sh "$patient"
done



