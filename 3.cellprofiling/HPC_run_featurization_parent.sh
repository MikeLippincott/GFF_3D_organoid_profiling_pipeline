#!/bin/bash

module load miniforge
conda init bash
conda activate GFF_featurization

PATIENT=$1
WELLFOV=$2
USEGPU=$3


git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

echo "Submitting jobs for $WELLFOV"
echo "Using GPU: $USEGPU"

number_of_jobs=$(squeue -u "$USER" | wc -l)
while [ "$number_of_jobs" -gt 950 ]; do
    sleep 1s
    number_of_jobs=$(squeue -u "$USER" | wc -l)
done

if [ "$USEGPU" = "TRUE" ]; then
    echo "Running GPU version"

    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=10:00 \
        --output=area_shape_gpu_child-%j.out \
        "$git_root"/3.cellprofiling/slurm_scripts/run_area_shape_child.sh "$PATIENT" "$WELLFOV" "$USEGPU"

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=30:00 \
        --output=colocalization_gpu_child-%j.out \
        "$git_root"/3.cellprofiling/slurm_scripts/run_colocalization_child.sh "$PATIENT" "$WELLFOV" "$USEGPU"

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=1:30:00 \
        --output=granularity_gpu_child-%j.out \
        "$git_root"/3.cellprofiling/slurm_scripts/run_granularity_child.sh "$PATIENT" "$WELLFOV" "$USEGPU"

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=3:00:00 \
        --output=intensity_gpu_child-%j.out \
        "$git_root"/3.cellprofiling/slurm_scripts/run_intensity_child.sh "$PATIENT" "$WELLFOV" "$USEGPU"

else
    echo "Running CPU version"

    sbatch \
        --nodes=1 \
        --ntasks=20 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=1:00:00 \
        --output=area_shape_cpu_child-%j.out \
        "$git_root"/3.cellprofiling/slurm_scripts/run_area_shape_child.sh "$PATIENT" "$WELLFOV" "$USEGPU"

    sbatch \
        --nodes=1 \
        --ntasks=25 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=1:00:00 \
        --output=colocalization_cpu_child-%j.out \
        "$git_root"/3.cellprofiling/slurm_scripts/run_colocalization_child.sh "$PATIENT" "$WELLFOV" "$USEGPU"

    sbatch \
        --nodes=1 \
        --ntasks=20 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=8:00:00 \
        --output=granularity_cpu_child-%j.out \
        "$git_root"/3.cellprofiling/slurm_scripts/run_granularity_child.sh "$PATIENT" "$WELLFOV" "$USEGPU"

    sbatch \
        --nodes=1 \
        --ntasks=25 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=6:00:00 \
        --output=intensity_cpu_child-%j.out \
        "$git_root"/3.cellprofiling/slurm_scripts/run_intensity_child.sh "$PATIENT" "$WELLFOV" "$USEGPU"

fi

sbatch \
    --nodes=1 \
    --ntasks=1 \
    --partition=amilan \
    --qos=normal \
    --account=amc-general \
    --time=10:00 \
    --output=neighbors_child-%j.out \
    "$git_root"/3.cellprofiling/slurm_scripts/run_neighbors_child.sh "$PATIENT" "$WELLFOV" "$USEGPU"



sbatch \
    --nodes=1 \
    --ntasks=20 \
    --partition=amilan \
    --qos=normal \
    --account=amc-general \
    --time=24:00:00 \
    --output=texture_child-%j.out \
    "$git_root"/3.cellprofiling/slurm_scripts/run_texture_child.sh "$PATIENT" "$WELLFOV" "$USEGPU"


echo "Featurization done"

