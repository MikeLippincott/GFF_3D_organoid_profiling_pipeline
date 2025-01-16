#!/bin/bash
conda activate GFF_cellprofiler

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

dir="../../data/cellprofiler/C4-2/"

python 0.run_cellprofiler_analysis.py --input_dir $dir

cd .. || exit

# deactivate cellprofiler environment
conda deactivate

echo "Cellprofiler analysis done"
