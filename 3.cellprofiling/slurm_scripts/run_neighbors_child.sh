#!/bin/bash

patient=$1
well_fov=$2

module load miniforge
conda init bash
conda activate GFF_featurization

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

echo "Running featurization for $patient $well_fov"

# start the timer
start_timestamp=$(date +%s)
python "$git_root"/3.cellprofiling/scripts/neighbors.py --patient "$patient" --well_fov "$well_fov" --processor_type "CPU"
end=$(date +%s)
echo "Time taken to run the featurization: (($end-$start_timestamp))"

conda deactivate
