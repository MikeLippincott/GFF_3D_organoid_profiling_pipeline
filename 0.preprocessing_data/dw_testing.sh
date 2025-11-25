#!/bin/bash


# resxy 130 pixel size (nm)
# resz 250 z-step size (nm)
# NA 1.46 numerical aperture
# ni 1.518 refractive index
# lambda 460 wavelength (nm)

dw_bw \
--resxy 100 \
--resz 1000 \
--NA 1.35 \
--ni 1.518 \
--lambda 461 \
PSF.tiff

example_image="/home/lippincm/Documents/GFF_3D_organoid_profiling_pipeline/data/NF0014_T1/zstack_images/C4-2/C4-2_405.tif"
# Deconvolve image.tiff -> dw_image.tiff
dw --iter 10 "$example_image" PSF.tiff --gpu --overwrite
