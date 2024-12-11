#!/bin/bash

NOTEBOOK=False

# activate  cellprofiler environment
conda activate GFF_cellpose

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

if [ "$NOTEBOOK" = True ]; then
    cd notebooks/ || exit
    papermill 0.segment_nuclei_organoids.ipynb 0.segment_nuclei_organoids.ipynb
    papermill 1.segment_cells_organoids.ipynb 1.segment_cells_organoids.ipynb
    papermill 3.make_segmentation_videos.ipynb 3.make_segmentation_videos.ipynb
    cd ../ || exit
else
    cd scripts/ || exit
    python 0.segment_nuclei_organoids.py --input_dir ../examples/raw_z_input/ --window_size 3 --clip_limit 0.05
    python 1.segment_cells_organoids.py --input_dir ../examples/raw_z_input/ --window_size 3 --clip_limit 0.1
    python 3.make_segmentation_videos.py
    cd ../ || exit
fi

# deactivate cellprofiler environment
conda deactivate

echo "Segmentation complete"
