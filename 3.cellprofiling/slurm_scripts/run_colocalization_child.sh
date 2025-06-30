#!/bin/bash
patient=$1
well_fov=$2
use_GPU=$3


echo "Running featurization for $patient $well_fov"
module load miniforge
conda init bash
conda activate GFF_featurization

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

# start the timer
start_timestamp=$(date +%s)
if [ "$use_GPU" = "TRUE" ]; then
    echo "Running GPU version"
    python "$git_root"/3.cellprofiling/scripts/colocalization_gpu.py --patient "$patient" --well_fov "$well_fov" --processor_type "GPU"
else
    echo "Running CPU version"
    python "$git_root"/3.cellprofiling/scripts/colocalization.py --patient "$patient" --well_fov "$well_fov" --processor_type "CPU"
fi

end=$(date +%s)
echo "Time taken to run the featurization: (($end-$start_timestamp))"

conda deactivate
