#!/bin/bash

patient=$1
well_fov=$2
compartment=$3
channel=$4
feature=$5
use_GPU=$6

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

echo "Patient: $patient, WellFOV: $well_fov, Feature: $feature, Compartment: $compartment, Channel: $channel, UseGPU: $use_GPU"

if [ "$use_GPU" == "both" ]; then
    processor_array=( "GPU" "CPU" )
else
    processor_array=( "$use_GPU" )
fi

# regardless of the processor type, texture and neighbors features are run on CPU
if [ "$feature" == "Neighbors" ]; then

    sbatch \
    --nodes=1 \
    --ntasks=1 \
    --partition=amilan \
    --qos=normal \
    --account=amc-general \
    --time=5:00 \
    --output=neighbors_child-%j.out \
    "$git_root"/3.cellprofiling/slurm_scripts/run_neighbors_child.sh \
        "$patient" \
        "$well_fov" \
        "$compartment" \
        "$channel"
fi

if [ "$feature" == "Texture" ] ; then
    echo "Running texture feature extraction"
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=2:00:00 \
        --output=texture_child-%j.out \
        "$git_root"/3.cellprofiling/slurm_scripts/run_texture_child.sh \
            "$patient" \
            "$well_fov" \
            "$compartment" \
            "$channel"
fi

for processor_type in "${processor_array[@]}"; do
    echo "Running featurization for $patient $well_fov with processor type: $processor_type"

    # AreaSizeShape feature extraction
    if [ "$feature" == "AreaSizeShape" ] ; then
        if [ "$processor_type" == "CPU" ]; then
            echo "Running CPU version for AreaSizeShape"
            sbatch \
                --nodes=1 \
                --ntasks=1 \
                --partition=amilan \
                --qos=normal \
                --account=amc-general \
                --time=5:00 \
                --output=area_shape_cpu_child-%j.out \
                "$git_root"/3.cellprofiling/slurm_scripts/run_area_shape_child.sh \
                "$patient" \
                "$well_fov" \
                "$compartment" \
                "$channel" \
                "$processor_type"
        else
            echo "Running GPU version for AreaSizeShape"
            sbatch \
                --nodes=1 \
                --ntasks=1 \
                --partition=aa100 \
                --qos=normal \
                --gres=gpu:1 \
                --account=amc-general \
                --time=10:00 \
                --output=area_shape_gpu_child-%j.out \
                "$git_root"/3.cellprofiling/slurm_scripts/run_area_shape_child.sh \
                "$patient" \
                "$well_fov" \
                "$compartment" \
                "$channel" \
                "$processor_type"
        fi
    fi
    if [ "$feature" == "Colocalization" ] ; then
        if [ "$processor_type" == "CPU" ]; then
            echo "Running CPU version for Colocalization"
            sbatch \
                --nodes=1 \
                --ntasks=1 \
                --partition=amilan \
                --qos=normal \
                --account=amc-general \
                --time=5:00 \
                --output=colocalization_cpu_child-%j.out \
                "$git_root"/3.cellprofiling/slurm_scripts/run_colocalization_child.sh \
                "$patient" \
                "$well_fov" \
                "$compartment" \
                "$channel" \
                "$processor_type"
        else
            echo "Running GPU version for Colocalization"
            sbatch \
                --nodes=1 \
                --ntasks=2 \
                --partition=aa100 \
                --qos=normal \
                --gres=gpu:1 \
                --account=amc-general \
                --time=5:00 \
                --output=colocalization_gpu_child-%j.out \
                "$git_root"/3.cellprofiling/slurm_scripts/run_colocalization_child.sh \
                "$patient" \
                "$well_fov" \
                "$compartment" \
                "$channel" \
                "$processor_type"
        fi
    fi
    if [ "$feature" == "Granularity" ] ; then
        if [ "$processor_type" == "CPU" ]; then
            echo "Running CPU version for Granularity"
            sbatch \
                --nodes=1 \
                --ntasks=2 \
                --partition=amilan \
                --qos=normal \
                --account=amc-general \
                --time=20:00 \
                --output=granularity_cpu_child-%j.out \
                "$git_root"/3.cellprofiling/slurm_scripts/run_granularity_child.sh \
                "$patient" \
                "$well_fov" \
                "$compartment" \
                "$channel" \
                "$processor_type"
        else
            echo "Running GPU version for Granularity"
            sbatch \
                --nodes=1 \
                --ntasks=2 \
                --partition=aa100 \
                --qos=normal \
                --gres=gpu:1 \
                --account=amc-general \
                --time=20:00 \
                --output=granularity_gpu_child-%j.out \
                "$git_root"/3.cellprofiling/slurm_scripts/run_granularity_child.sh \
                "$patient" \
                "$well_fov" \
                "$compartment" \
                "$channel" \
                "$processor_type"
        fi
    fi

    if [ "$feature" == "Intensity" ] ; then
        if [ "$processor_type" == "CPU" ]; then
            echo "Running CPU version for Intensity"
            sbatch \
                --nodes=1 \
                --ntasks=2 \
                --partition=amilan \
                --qos=normal \
                --account=amc-general \
                --time=5:00 \
                --output=intensity_cpu_child-%j.out \
                "$git_root"/3.cellprofiling/slurm_scripts/run_intensity_child.sh \
                "$patient" \
                "$well_fov" \
                "$compartment" \
                "$channel" \
                "$processor_type"
        else
            echo "Running GPU version for Intensity"
            sbatch \
                --nodes=1 \
                --ntasks=2 \
                --partition=aa100 \
                --qos=normal \
                --gres=gpu:1 \
                --account=amc-general \
                --time=5:00 \
                --output=intensity_gpu_child-%j.out \
                "$git_root"/3.cellprofiling/slurm_scripts/run_intensity_child.sh \
                    "$patient" \
                    "$well_fov" \
                    "$compartment" \
                    "$channel" \
                    "$processor_type"
        fi
    fi
done

echo "Featurization done"

