#!/bin/bash

conda activate gff_preprocessing_env
git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

conda activate GFF_featurization

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

cd scripts/ || exit

GPU_OPTIONS=( "TRUE" "FALSE" )

patient="NF0014"
parent_dir="${git_root}/data/${patient}/zstack_images"
# get the list of all dirs in the parent_dir
dirs=$(ls -d "$parent_dir"/*)

for dir in $dirs; do
    well_fov=$(basename "$dir")
    for use_GPU in "${GPU_OPTIONS[@]}"; do
        echo "$well_fov"

        if [ "$use_GPU" = "TRUE" ]; then
            echo "Running GPU version"

            bash "$git_root"/3.cellprofiling/slurm_scripts/run_area_shape_child.sh "$patient" "$well_fov" "$use_GPU"

            bash "$git_root"/3.cellprofiling/slurm_scripts/run_colocalization_child.sh "$patient" "$well_fov" "$use_GPU"

            bash "$git_root"/3.cellprofiling/slurm_scripts/run_granularity_child.sh "$patient" "$well_fov" "$use_GPU"

            bash "$git_root"/3.cellprofiling/slurm_scripts/run_intensity_child.sh "$patient" "$well_fov" "$use_GPU"

        else
            echo "Running CPU version"

            bash "$git_root"/3.cellprofiling/slurm_scripts/run_area_shape_child.sh "$patient" "$well_fov" "$use_GPU"

            bash "$git_root"/3.cellprofiling/slurm_scripts/run_colocalization_child.sh "$patient" "$well_fov" "$use_GPU"

            bash "$git_root"/3.cellprofiling/slurm_scripts/run_granularity_child.sh "$patient" "$well_fov" "$use_GPU"

            bash "$git_root"/3.cellprofiling/slurm_scripts/run_intensity_child.sh "$patient" "$well_fov" "$use_GPU"
        fi
            bash "$git_root"/3.cellprofiling/slurm_scripts/run_neighbors_child.sh "$patient" "$well_fov" "$use_GPU"

            bash "$git_root"/3.cellprofiling/slurm_scripts/run_texture_child.sh "$patient" "$well_fov" "$use_GPU"
    done
done

echo "Featurization done"
