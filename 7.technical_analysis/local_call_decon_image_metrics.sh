#!/bin/bash
jupyter nbconvert --to script --output-dir=scripts/ notebooks/*.ipynb
cd scripts || exit 1
load_file_path="../loadfiles/decon_image_metrics_load_file.txt"
conda activate GFF_DL_featurization

while IFS= read -r line; do
    # split the line into an array
    IFS=$'\t' read -r -a parts <<< "$line"
    # assign the parts to variables
    patient="${parts[0]}"
    well_fov="${parts[1]}"
    echo "Submitting decon image metrics for Patient: $patient, WellFOV: $well_fov"


    python decon_image_metrics.py --patient "$patient" --well_fov "$well_fov"


done < "$load_file_path"
conda deactivate
cd .. || exit 1
echo "All segmentation child jobs submitted"

