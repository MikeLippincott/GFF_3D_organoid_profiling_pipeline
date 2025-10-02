#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import sys

import matplotlib.pyplot as plt

# Import dependencies
import numpy as np
import pandas as pd
import skimage
import tifffile
from skimage import io

cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd
else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break
sys.path.append(str(root_dir / "utils"))
from arg_parsing_utils import check_for_missing_args, parse_args
from file_reading import read_zstack_image
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()
if in_notebook:
    from tqdm.notebook import tqdm
else:
    import tqdm

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


if not in_notebook:
    args = parse_args()
    window_size = args["window_size"]
    clip_limit = args["clip_limit"]
    well_fov = args["well_fov"]
    patient = args["patient"]
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
        window_size=window_size,
        clip_limit=clip_limit,
    )
else:
    print("Running in a notebook")
    patient = "NF0014_T1"
    well_fov = "C4-2"
    window_size = 3
    clip_limit = 0.05


input_dir_raw = pathlib.Path(
    f"{image_base_dir}/data/{patient}/zstack_images/{well_fov}"
).resolve(strict=True)
input_dir_decon = pathlib.Path(
    f"{image_base_dir}/data/{patient}/deconvolved_images/{well_fov}"
).resolve(strict=True)


# In[3]:


# read in the image
list_of_decon_images = list(input_dir_decon.glob("*.tif"))
list_of_raw_images = list(input_dir_raw.glob("*.tif"))
list_of_decon_images.sort()
list_of_raw_images.sort()


# In[4]:


convolutions = 25
convolution_step = 1


# In[6]:


img_dict = {
    "decon": list_of_decon_images,
    "raw": list_of_raw_images,
}
for convolution in range(1, convolutions + 1):
    img_dict[f"convolved_{convolution}"] = []
for img_path in tqdm(list_of_decon_images, desc="Processing image set"):
    img = read_zstack_image(img_path)
    for convolution in tqdm(
        range(1, convolutions + 1), desc="Convolving image", leave=False
    ):
        img = skimage.filters.gaussian(img, sigma=3)
        if (convolution) % convolution_step == 0:
            convolution_output_path = pathlib.Path(
                f"{image_base_dir}/data/{patient}/convolution_{convolution}/{well_fov}"
            ).resolve()
            convolution_output_path.mkdir(parents=True, exist_ok=True)
            if not (convolution_output_path / f"{img_path.name}").exists():
                tifffile.imwrite(
                    pathlib.Path(convolution_output_path / f"{img_path.name}"),
                    img,
                )
            if convolution % 5 == 0:
                img_dict[f"convolved_{convolution}"].append(
                    convolution_output_path / f"{img_path.name}"
                )
