#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib
import shutil
import sys

import numpy as np
import tqdm

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
    # check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")

sys.path.append(str(pathlib.Path(f"{root_dir}/utils").resolve()))
from file_checking import check_number_of_files

# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser(
        description="set up directories for the analysis of the data"
    )

    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="patient name, e.g. 'P01'",
    )
    argparser.add_argument(
        "--well_fov",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = argparser.parse_args()
    patient = args.patient
    well_fov = args.well_fov
else:
    patient = "NF0014"
    well_fov = "C4-2"


# In[3]:


# set path to the processed data dir
segmentation_data_dir = pathlib.Path(
    f"{root_dir}/data/{patient}/segmentation_masks/{well_fov}"
).resolve(strict=True)
zstack_dir = pathlib.Path(
    f"{root_dir}/data/{patient}/zstack_images/{well_fov}"
).resolve(strict=True)


# In[4]:


# perform checks for each directory
segmentation_data_files = list(segmentation_data_dir.glob("*"))
segmentation_data_files


# ## Copy files from processed dir to cellprofiler images dir

# In[5]:


masks_names_to_keep = [
    "cell_masks_watershed.tiff",
    "cytoplasm_mask.tiff",
    "nuclei_masks_reassigned.tiff",
    "organoid_masks_reconstructed.tiff",
]


# In[6]:


# remove files that are not in the list of masks to keep
for file in tqdm.tqdm(segmentation_data_files):
    if file.name not in masks_names_to_keep:
        file.unlink()
        print(f"Removed file: {file.name}")


# In[7]:


# copy the masks to the zstack directory
for file in tqdm.tqdm(segmentation_data_files):
    if file.name in masks_names_to_keep:
        destination = zstack_dir / file.name
        if not destination.exists():
            shutil.copy(file, destination)
            print(f"Copied file: {file.name} to {destination}")
        else:
            print(f"File {file.name} already exists in {zstack_dir}, skipping copy.")
