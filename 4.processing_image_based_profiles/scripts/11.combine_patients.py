#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import duckdb
import pandas as pd

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

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


patient_ids_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True)
patients = pd.read_csv(patient_ids_path, header=None, names=["patient_id"], dtype=str)[
    "patient_id"
].to_list()


# Merge patients by the following levels:

# In[3]:


levels_to_merge_dict = {
    "norm": {
        "sc": [],
        "organoid": [],
    },
    "fs": {
        "sc_fs": [],
        "organoid_fs": [],
    },
    "agg": {
        "sc_agg_parent_organoid_level": [],
        "sc_agg_well_level": [],
        "sc_consensus": [],
        "organoid_agg_well_level": [],
        "organoid_consensus": [],
    },
    "merged": {
        "sc-organoid_sc_agg_well_parent_organoid_level": [],
        "sc-organoid_agg_well_level": [],
        "sc-organoid_consensus": [],
    },
}


# In[4]:


for patient in patients:
    norm_path = pathlib.Path(
        f"{root_dir}/data/{patient}/image_based_profiles/3.normalized_profiles"
    )
    fs_path = pathlib.Path(
        f"{root_dir}/data/{patient}/image_based_profiles/4.feature_selected_profiles"
    )
    agg_path = pathlib.Path(
        f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles"
    )
    merge_path = pathlib.Path(
        f"{root_dir}/data/{patient}/image_based_profiles/6.merged_profiles"
    )
    for file in norm_path.glob("*.parquet"):
        if "sc" in file.name:
            levels_to_merge_dict["norm"]["sc"].append(file)
        elif "organoid" in file.name:
            levels_to_merge_dict["norm"]["organoid"].append(file)
    for file in fs_path.glob("*.parquet"):
        if "sc" in file.name:
            levels_to_merge_dict["fs"]["sc_fs"].append(file)
        elif "organoid" in file.name:
            levels_to_merge_dict["fs"]["organoid_fs"].append(file)
    for file in agg_path.glob("*.parquet"):
        for key in levels_to_merge_dict["agg"].keys():
            if key in file.name:
                levels_to_merge_dict["agg"][key].append(file)
    for file in merge_path.glob("*.parquet"):
        for key in levels_to_merge_dict["merged"].keys():
            if key in file.name:
                levels_to_merge_dict["merged"][key].append(file)


# In[5]:


for level, files_dict in levels_to_merge_dict.items():
    for profile_type, files in files_dict.items():
        if not files:
            continue

        # Read and merge the parquet files
        df = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
        print(f"Merged df shape for {level} - {profile_type}: {df.shape}")

        # Optionally, you can save the merged table to a parquet file
        output_path = pathlib.Path(
            f"{root_dir}/data/all_patient_IBPs/{level}_{profile_type}_merged.parquet"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
