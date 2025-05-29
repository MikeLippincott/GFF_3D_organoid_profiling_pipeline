#!/bin/bash


# activate cellprofiler environment
# The following environment activation commands are commented out.
# Ensure the required environment is activated manually before running this script,
# or confirm that activation is handled by a parent script or workflow.
# module load anaconda
# conda init bash
# conda activate GFF_segmentation

cd scripts/ || exit

well_fov=$1
patient=$2
echo "Processing well_fov $well_fov for patient $patient"
compartments=( "nuclei" "organoid" ) # we do not do 2.5D segmentation for cells in this script


python 0.segment_nuclei_organoids.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --window_size 3 \
    --clip_limit 0.05

python 2.segment_whole_organoids.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --window_size 4 \
    --clip_limit 0.1

conda deactivate
conda activate GFF_segmentation_cellpose4
python 1.segment_cells_watershed_method.ipynb \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --clip_limit 0.05


for compartment in "${compartments[@]}"; do

    if [ "$compartment" == "nuclei" ]; then
        window_size=3
    elif [ "$compartment" == "organoid" ]; then
        window_size=4
    else
        echo "Not specified compartment: $compartment"

    fi
    python 3.segmentation_decoupling.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment" \
        --window_size "$window_size"

    python 4.reconstruct_3D_masks_copy.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment"

    python 5.post-hoc_mask_refinement.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment"
done

python 5.post-hoc_mask_refinement.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --compartment "cell"

python 6.post_hoc_reassignment.py \
    --patient "$patient" \
    --well_fov "$well_fov"

python 7.create_cytoplasm_masks.py \
    --patient "$patient" \
    --well_fov "$well_fov"


cd ../ || exit

# deactivate cellprofiler environment
# conda deactivate

echo "Segmentation complete"
