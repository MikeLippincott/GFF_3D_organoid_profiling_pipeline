#!/bin/bash

module load miniforge
conda init bash
conda activate GFF_featurization

cd ../scripts/ || exit
well_fov=$1
patient=$3

echo "Running featurization for $patient $well_fov"

# start the timer
start_timestamp=$(date +%s)
python neighbors.py --well_fov $well_fov --patient $patient
end=$(date +%s)
echo "Time taken to run the featurization: $(($end-$start_timestamp))"

cd ../ || exit

conda deactivate
