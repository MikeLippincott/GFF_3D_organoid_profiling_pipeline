#!/bin/bash

conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

GPU=False

# start the timer
start=$(timestamp)
# parent_dir="../../data/NF0014/cellprofiler"
# # get the list of all dirs in the parent_dir
# dirs=$(ls -d $parent_dir/*)

# for dir in $dirs; do
#     dir_name=$(basename $dir)
#     echo $dir_name
#         # python run_featurization.py --input_dir $dir
# done

if [ "$GPU" = True ]; then
    echo "Running GPU version"
    python area_shape_gpu.py
    python colocalization_gpu.py
    python granularity_gpu.py
    python instensity_gpu.py
    python neighbors.py
    python texture.py
else
    echo "Running CPU version"
    python area_shape.py
    python colocalization.py
    python granularity.py
    python instensity.py
    python neighbors.py
    python texture.py
fi

end=$(timestamp)
echo "Time taken to run the featurization: $(get_duration $start $end)"

cd ../ || exit

echo "Featurization done"
