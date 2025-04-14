#!/bin/bash

use_GPU=$1

module load miniforge
conda init bash
conda activate GFF_featurization

cd ../scripts/ || exit

# start the timer
start_timestamp=$(date +%s)
if [ "$use_GPU" = "TRUE" ]; then
    echo "Running GPU version"
    python area_shape_gpu.py
else
    echo "Running CPU version"
    python area_shape.py
fi

end=$(date +%s)
echo "Time taken to run the featurization: $(($end-$start_timestamp))"

cd ../ || exit

conda deactivate
