#!/bin/bash


# activate  cellprofiler environment
conda activate GFF_cellpose

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

python 0.segment_nuclei_organoids.py --input_dir ../test_dir/ --window_size 3 --clip_limit 0.05

cd ../ || exit

# deactivate cellprofiler environment
conda deactivate

echo "Test complete"
