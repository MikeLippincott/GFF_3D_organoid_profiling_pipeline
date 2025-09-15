#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

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

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[ ]:


if not in_notebook:
    args = parse_args()
    well_fov = args["well_fov"]
    patient = args["patient"]
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
    )

else:
    print("Running in a notebook")
    well_fov = "C4-2"
    patient = "NF0014_T1"


input_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/deconvolved_images/{well_fov}/"
    # f"{image_base_dir}/data/{patient}/zstack_images/{well_fov}"
).resolve(strict=True)
mask_input_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/deconvolved_segmentation_masks/{well_fov}"
    # f"{image_base_dir}/data/{patient}/segmentation_masks/{well_fov}"
).resolve(strict=True)
output_file_path = pathlib.Path(mask_input_dir / "cytoplasm_mask.tiff").resolve()


# In[9]:


# get all the masks
nuclei_masks_path = pathlib.Path(
    mask_input_dir / "nuclei_masks_reconstructed_corrected.tiff"
).resolve(strict=True)
cell_masks_path = pathlib.Path(mask_input_dir / "cell_masks_watershed.tiff").resolve(
    strict=True
)

nuclei_masks = read_zstack_image(nuclei_masks_path)
cell_masks = read_zstack_image(cell_masks_path)


# In[10]:


cytoplasm_masks = np.zeros_like(cell_masks)
# filter masks that are not the background
for z_slice_index in range(nuclei_masks.shape[0]):
    nuclei_slice_mask = nuclei_masks[z_slice_index]
    cell_slice_mask = cell_masks[z_slice_index]
    cytoplasm_mask = cell_slice_mask.copy()
    cytoplasm_mask[nuclei_slice_mask > 0] = 0
    cytoplasm_masks[z_slice_index] = cytoplasm_mask


# In[11]:


tifffile.imwrite(output_file_path, cytoplasm_masks)


# In[12]:


if in_notebook:
    for z in range(nuclei_masks.shape[0]):
        plt.subplot(1, 3, 1)
        plt.imshow(nuclei_masks[z])
        plt.title("Nuclei Mask")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(cell_masks[z])
        plt.title("Cell Mask")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(cytoplasm_masks[z])
        plt.title("Cytoplasm Mask")
        plt.axis("off")
        plt.show()
