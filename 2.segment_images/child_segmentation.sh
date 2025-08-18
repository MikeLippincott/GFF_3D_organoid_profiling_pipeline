#!/bin/bash


# activate cellprofiler environment
# The following environment activation commands are commented out.
# Ensure the required environment is activated manually before running this script,
# or confirm that activation is handled by a parent script or workflow.
# find the git repository root directory
module load anaconda
conda init bash
conda activate GFF_segmentation

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

patient=$1
well_fov=$2
echo "Processing well_fov $well_fov for patient $patient"
compartments=( "nuclei" "organoid" ) # we do not do 2.5D segmentation for cells in this script
# cells get segmented using a 3D method rather a 2.5D method

python "$git_root"/2.segment_images/scripts/0.segment_nuclei.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --window_size 3 \
    --clip_limit 0.05

python "$git_root"/2.segment_images/scripts/1.segment_whole_organoids.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --window_size 4 \
    --clip_limit 0.1

for compartment in "${compartments[@]}"; do

    if [ "$compartment" == "nuclei" ]; then
        window_size=3
    elif [ "$compartment" == "organoid" ]; then
        window_size=4
    else
        echo "Not specified compartment: $compartment"

    fi
    python "$git_root"/2.segment_images/scripts/2.segmentation_decoupling.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment" \
        --window_size "$window_size"

    python "$git_root"/2.segment_images/scripts/3.reconstruct_3D_masks.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment"

    python "$git_root"/2.segment_images/scripts/4.post-hoc_mask_refinement.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment"
done

python "$git_root"/2.segment_images/scripts/5.segment_cells_watershed_method.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --clip_limit 0.05

python "$git_root"/2.segment_images/scripts/4.post-hoc_mask_refinement.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --compartment "cell"

python "$git_root"/2.segment_images/scripts/6.post-hoc_reassignment.py \
    --patient "$patient" \
    --well_fov "$well_fov"

python "$git_root"/2.segment_images/scripts/7.create_cytoplasm_masks.py \
    --patient "$patient" \
    --well_fov "$well_fov"

python "$git_root"/2.segment_images/scripts/9.clean_up_segmentation.py \
    --patient "$patient" \
    --well_fov "$well_fov"

conda deactivate

echo "Segmentation completed for well_fov $well_fov and patient $patient"
