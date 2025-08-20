#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import sys

import numpy as np
import tqdm

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
from notebook_init_utils import bandicoot_check, init_notebook
from segmentation_init_utils import parse_segmentation_args

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path("/home/lippincm/mnt/bandicoot").resolve(), root_dir
)

sys.path.append(str(pathlib.Path(f"{root_dir}/utils").resolve()))
from file_checking import check_number_of_files

# In[ ]:


if not in_notebook:
    args = parse_segmentation_args()
    patient = args["patient"]
else:
    patient = "NF0014"


# In[ ]:


# set path to the processed data dir
segmentation_data_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/segmentation_masks/"
).resolve(strict=True)
zstack_dir = pathlib.Path(f"{image_base_dir}/data/{patient}/zstack_images/").resolve(
    strict=True
)


# In[ ]:


# perform checks for each directory
segmentation_data_dirs = list(segmentation_data_dir.glob("*"))
segmentation_data_dirs = [d for d in segmentation_data_dirs if d.is_dir()]
zstack_dirs = list(zstack_dir.glob("*"))
zstack_dirs = [d for d in zstack_dirs if d.is_dir()]

print("Checking segmentation data directories")
for dir in segmentation_data_dirs:
    check_number_of_files(dir, 4)

print("Checking zstack directories")
for dir in zstack_dirs:
    check_number_of_files(dir, 9)
