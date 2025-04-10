#!/bin/bash

use_GPU=$1

module load miniforge
conda init bash
conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

# start the timer
start_timestamp=$(date +%s)
if [ "$use_GPU" = "TRUE" ]; then
    echo "Running GPU version"
    python granularity_gpu.py
else
    echo "Running CPU version"
    python granularity.py
fi

end=$(date +%s)
echo "Time taken to run the featurization: $(($end-$start_timestamp))"

cd ../ || exit

conda deactivate
