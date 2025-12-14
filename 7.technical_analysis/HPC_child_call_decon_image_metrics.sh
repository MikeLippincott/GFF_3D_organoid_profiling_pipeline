#!/bin/bash

module load miniforge
conda init bash
conda activate GFF_DL_featurization

patient=$1
well_fov=$2

cd scripts || exit 1

echo "Processing patient $patient - well_fov $well_fov"

python decon_image_metrics.py --patient "$patient" --well_fov "$well_fov"

# deactivate cellprofiler environment
conda deactivate
cd .. || exit 1

echo "All segmentation child jobs submitted"
