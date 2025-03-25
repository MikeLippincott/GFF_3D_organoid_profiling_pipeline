#!/usr/bin/env python
# coding: utf-8

# This notebook reslices the 3D volume into interpolated slices strictly for the purposes of visualization and segmentation.

# In[1]:


import argparse
import pathlib

import matplotlib.pyplot as plt

# Import dependencies
import numpy as np
import skimage
import tifffile
import tqdm
from skimage import io

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[ ]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)
else:
    print("Running in a notebook")
    input_dir = pathlib.Path("../../data/NF0014/zstack_images/C4-2/").resolve(
        strict=True
    )


output_dir = pathlib.Path(
    f"../../data/NF0014/resliced_images/{input_dir.stem}"
).resolve()
output_dir.mkdir(exist_ok=True, parents=True)
mask_path = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve()
mask_path.mkdir(exist_ok=True, parents=True)


# ## Set up images, paths and functions

# In[3]:


image_extensions = {".tif", ".tiff"}
files = sorted(input_dir.glob("*"))
files = [str(x) for x in files if x.suffix in image_extensions]


# In[ ]:


# reslice the image such that the z-distance between slices is the same as the pixel size
# of the original image
pixel_size = 0.1
z_distance = 1
reslice_factor = z_distance / pixel_size
for f in tqdm.tqdm(files):
    imgs = io.imread(f)
    imgs = skimage.transform.rescale(imgs, (reslice_factor, 1, 1), anti_aliasing=True)
    print("Resliced image shape:", imgs.shape)
    tifffile.imwrite(
        output_dir / f"{input_dir.stem}_{pathlib.Path(f).stem.split('_')[1]}.tif", imgs
    )
