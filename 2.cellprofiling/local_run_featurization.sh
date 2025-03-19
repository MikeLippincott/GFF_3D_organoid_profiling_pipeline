#!/bin/bash

conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

parent_dir="../../data/NF0014/cellprofiler"
# get the list of all dirs in the parent_dir
dirs=$(ls -d $parent_dir/*)

for dir in $dirs; do
    dir_name=$(basename $dir)
    echo $dir_name
        python run_featurization.py --input_dir $dir
done


cd ../ || exit

echo "Featurization done"
