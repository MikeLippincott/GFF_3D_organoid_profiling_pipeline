#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import tqdm

sys.path.append(str(pathlib.Path("../../utils").resolve()))
from file_checking import check_number_of_files

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


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
        "--overwrite",
        action="store_true",
        help="overwrite existing directories",
    )

    args = argparser.parse_args()
    patient = args.patient
else:
    patient = "NF0014"
    overwrite = False


# In[3]:


features_path = pathlib.Path(f"../../data/{patient}/extracted_features/").resolve(
    strict=True
)


# In[4]:


investigate_further = []


# In[5]:


well_fovs = [x for x in features_path.iterdir() if x.is_dir()]
well_fovs = sorted(well_fovs)
for dir in tqdm.tqdm(well_fovs):
    if "stats" in dir.name:
        check_number_of_files(dir, ((len(well_fovs) - 1) * 6))
    else:
        if not check_number_of_files(dir, 105):
            investigate_further.append(dir)
print(f"Found {len(investigate_further)} directories that are not complete...")


# In[6]:


rerun_dict = {
    "AreaSizeShape": [],
    "Colocalization": [],
    "Granularity": [],
    "Intensity": [],
    "Neighbors": [],
    "Texture": [],
}


# In[7]:


files = {
    "patient": [],
    "well_fov": [],
    "file": [],
}
for dir in investigate_further:
    # get all files in the directory
    files_in_dir = sorted(dir.glob("*"))
    for file in files_in_dir:
        if file.is_file():
            files["patient"].append(patient)
            files["well_fov"].append(dir.name)
            files["file"].append(file.name)
df = pd.DataFrame(files)
df["Type"] = df["file"].apply(lambda x: x.split("_")[0])
df = df.groupby(["well_fov", "Type"]).count().reset_index()


# In[8]:


# filter for rows that have AreaSize < 4
rerun_dict["AreaSizeShape"] = df[(df["file"] < 4) & (df["Type"] == "AreaSizeShape")][
    "well_fov"
].to_list()
rerun_dict["Colocalization"] = df[(df["file"] < 40) & (df["Type"] == "Colocalization")][
    "well_fov"
].to_list()
rerun_dict["Granularity"] = df[(df["file"] < 20) & (df["Type"] == "Granularity")][
    "well_fov"
].to_list()
rerun_dict["Intensity"] = df[(df["file"] < 20) & (df["Type"] == "Intensity")][
    "well_fov"
].to_list()
rerun_dict["Neighbors"] = df[(df["file"] < 1) & (df["Type"] == "Neighbors")][
    "well_fov"
].to_list()


# In[9]:


rerun_dict
