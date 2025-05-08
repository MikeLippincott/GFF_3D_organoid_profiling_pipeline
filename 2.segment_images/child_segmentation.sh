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
compartments=( "nuclei" "cell" "organoid" )

# echo "Reslicing images" # comment out for now -> needs further testing
# python 00.reslice_images.py --well_fov "$well_fov"
echo "Segmenting Nuclei"
python 0.segment_nuclei_organoids.py --patient "$patient" --well_fov "$well_fov" --window_size 2 --clip_limit 0.05 >> segmentation.log
echo "Completed Nuclei Segmentation"
echo "Segmenting Cells"
python 1.segment_cells_organoids.py --patient "$patient" --well_fov "$well_fov" --window_size 3 --clip_limit 0.1 >> segmentation.log
echo "Completed Cell Segmentation"
echo "Segmenting Organoids"
python 2.segment_whole_organoids.py --patient "$patient" --well_fov "$well_fov" --window_size 4 --clip_limit 0.1 >> segmentation.log
echo "Completed Organoid Segmentation"
for compartment in "${compartments[@]}"; do
    echo "Decoupling $compartment"
    python 3.segmentation_decoupling.py --patient "$patient" --well_fov "$well_fov" --compartment "$compartment" >> segmentation.log
    python 4.reconstruct_3D_masks.py --patient "$patient" --well_fov "$well_fov" --compartment "$compartment" --radius_constraint 10 >> segmentation.log
done
python 5.create_cytoplasm_masks.py --patient "$patient" --well_fov "$well_fov" >> segmentation.log

cd ../ || exit

# deactivate cellprofiler environment
# conda deactivate

echo "Segmentation complete"
