#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as io
import tifffile

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
        "--well_fov",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    well_fov = args.well_fov
    mask_input_dir = pathlib.Path(f"../processed_data/{well_fov}").resolve(strict=True)
else:
    print("Running in a notebook")
    well_fov = "C4-2"
    mask_input_dir = pathlib.Path(f"../processed_data/{well_fov}").resolve(strict=True)

base_input_dir = "../../data/NF0014/processed_data/"
input_dir = pathlib.Path(f"{base_input_dir}/{well_fov}").resolve(strict=True)
mask_path = pathlib.Path(f"../../data/NF0014/processed_data/{well_fov}").resolve()
mask_path.mkdir(parents=True, exist_ok=True)
output_file_path = pathlib.Path(mask_path / "cytoplasm_mask.tiff").resolve()


# In[3]:


# get all the masks
nuclei_masks_path = pathlib.Path(mask_input_dir / "nuclei_masks.tiff").resolve(
    strict=True
)
cell_masks_path = pathlib.Path(mask_input_dir / "cell_masks.tiff").resolve(strict=True)

nuclei_masks = io.imread(nuclei_masks_path)
cell_masks = io.imread(cell_masks_path)


# In[ ]:


cytoplasm_masks = np.zeros_like(cell_masks)
# filter masks that are not the background
for z_slice_index in range(nuclei_masks.shape[0]):
    nuclei_slice_mask = nuclei_masks[z_slice_index]
    cell_slice_mask = cell_masks[z_slice_index]
    cytoplasm_mask = cell_slice_mask.copy()
    cytoplasm_mask[nuclei_slice_mask > 0] = 0
    cytoplasm_masks[z_slice_index] = cytoplasm_mask


# In[ ]:


tifffile.imwrite(output_file_path, cytoplasm_masks)


# In[6]:


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
