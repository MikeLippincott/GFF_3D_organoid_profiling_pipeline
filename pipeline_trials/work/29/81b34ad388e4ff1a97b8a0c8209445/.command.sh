#!/bin/bash -ue
cd /home/lippincm/Documents/GFF_3D_organoid_profiling_pipeline/pipeline_trials/../2.segment_images/ || exit 1
echo "Processing patient: NF0014, well_fov: C4-2"
source child_segmentation.sh C4-2 NF0014
cd /home/lippincm/Documents/GFF_3D_organoid_profiling_pipeline/pipeline_trials/ || exit 1
