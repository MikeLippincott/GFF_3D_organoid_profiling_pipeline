#!/bin/bash

conda activate gff_preprocessing_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# python 0.update_file_structure.py
python 1.make_z-stack_images.py
python 2.make_z-stack_images.py


cd .. || exit

conda deactivate

echo "Preprocessing complete"
