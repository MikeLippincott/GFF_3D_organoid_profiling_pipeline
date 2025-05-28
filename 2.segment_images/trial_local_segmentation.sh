#!/bin/bash

# activate  cellprofiler environment
conda init bash
conda activate GFF_segmentation_cellpose3

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


cd scripts/ || exit
patient="NF0014"
well_fov="C2-2"


# # get all input directories in specified directory
compartments=( "nuclei" "cell" "organoid" )

python 0.segment_nuclei_organoids.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --window_size 3 \
    --clip_limit 0.05

python 2.segment_whole_organoids.py --patient "$patient" --well_fov "$well_fov" --window_size 4 --clip_limit 0.1

conda deactivate
conda activate GFF_segmentation_cellpose4
python 1.segment_cells_organoids.ipynb \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --window_size 4 \
    --clip_limit 0.05


for compartment in "${compartments[@]}"; do

    if [ "$compartment" == "nuclei" ]; then
        window_size=3
    elif [ "$compartment" == "cell" ]; then
        window_size=4
    elif [ "$compartment" == "organoid" ]; then
        window_size=3
    else
        echo "Unknown compartment: $compartment"
        exit 1
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

    python 8.post-hoc_correction_in_loop.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment"

done


conda deactivate

conda activate viz_env

python visualize_segmentation.py --well_fov "$well_fov"

conda deactivate

cd ../ || exit

echo "Segmentation complete"
