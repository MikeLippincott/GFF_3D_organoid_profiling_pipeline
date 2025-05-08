#!/bin/bash -ue
cd /home/lippincm/Documents/GFF_3D_organoid_profiling_pipeline/pipeline_trials/../3.cellprofiling/slurm_scripts/ || exit 1
echo "Running GPU featurization for patient: NF0014, well_fov: C4-2 use_gpu: false"
bash run_neighbors_child.sh C4-2 FALSE NF0014
cd /home/lippincm/Documents/GFF_3D_organoid_profiling_pipeline/pipeline_trials/ || exit 1
