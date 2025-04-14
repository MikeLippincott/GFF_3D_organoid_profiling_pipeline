#!/bin/bash

module load miniforge
conda init bash
conda activate GFF_featurization

cd ../scripts/ || exit

# start the timer
start_timestamp=$(date +%s)
python neighbors.py
end=$(date +%s)
echo "Time taken to run the featurization: $(($end-$start_timestamp))"

cd ../ || exit

conda deactivate
