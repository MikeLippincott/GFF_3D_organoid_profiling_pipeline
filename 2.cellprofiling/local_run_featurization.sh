#!/bin/bash

conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

GPU=False

# start the timer
start_timestamp=$(date +%s)


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
    time python area_shape_gpu.py
    time python colocalization_gpu.py
    time python granularity_gpu.py
    time python instensity_gpu.py
    time python neighbors.py
    time python texture.py
else
    echo "Running CPU version"
    # time python area_shape.py
    time python colocalization.py
    time python granularity.py
    time python instensity.py
    time python neighbors.py
    time python texture.py
fi

end=$(date +%s)
echo "Time taken to run the featurization: $(($end-$start_timestamp))"

cd ../ || exit

echo "Featurization done"
