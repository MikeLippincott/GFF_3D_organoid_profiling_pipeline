#!/bin/bash -ue
cd /home/lippincm/Documents/GFF_3D_organoid_profiling_pipeline/pipeline_trials/../3.cellprofiling/slurm_scripts/ || exit 1
echo "Processing patient: NF0014, well_fov: C4-2"
bash run_colocalization_child.sh NF0014 FALSE C4-2
cd /home/lippincm/Documents/GFF_3D_organoid_profiling_pipeline/pipeline_trials/ || exit 1
