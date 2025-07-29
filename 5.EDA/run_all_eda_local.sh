#!/bin/bash

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

conda activate GFF_analysis

jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/5.EDA/scripts/ "$git_root"/5.EDA/notebooks/*.ipynb

# python "$git_root"/5.EDA/scripts/0.generate_umap.py

conda deactivate
conda activate gff_figure_env

# Rscript "$git_root"/5.EDA/scripts/1.plot_umap.R

patient_array_file_path="$git_root/data/patient_IDs.txt"
# read the patient IDs from the file into an array
if [[ -f "$patient_array_file_path" ]]; then
    readarray -t patient_array < "$patient_array_file_path"
else
    echo "Error: File $patient_array_file_path does not exist."
    exit 1
fi

for patient in "${patient_array[@]}"; do
    echo "Processing patient: $patient"
    Rscript "$git_root"/5.EDA/scripts/2.consensus_profiles.r --patient "$patient"
done

conda deactivate

echo "All scripts executed successfully."
