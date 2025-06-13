#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile feature selection.

# In[1]:


import argparse
import pathlib

import pandas as pd
from pycytominer import feature_select

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID to process, e.g. 'P01'",
    )
    args = argparser.parse_args()
    patient = args.patient

else:
    patient = "NF0014"


# In[3]:


# pathing
sc_normalized_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/2.sc_normalized_profiles.parquet"
).resolve(strict=True)
organoid_normalized_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/2.organoid_normalized_profiles.parquet"
).resolve(strict=True)


# output path
sc_fs_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/3.sc_fs_profiles.parquet"
).resolve()
organoid_fs_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/3.organoid_fs_profiles.parquet"
).resolve()


# In[4]:


# read in the data
sc_normalized = pd.read_parquet(sc_normalized_path)
organoid_normalized = pd.read_parquet(organoid_normalized_path)


# In[5]:


feature_select_ops = [
    "variance_threshold",
    "drop_na_columns",
    "correlation_threshold",
    "blocklist",
]


# ### Feature select the single-cell profiles

# In[6]:


sc_normalized.head()


# In[7]:


sc_blocklist = [
    x
    for x in sc_normalized.columns
    if "Area" in x and ("MAX" in x or "MIN" in x or "BBOX" in x or "CENTER" in x)
]
# write the blocklist to a file
# add "blocklist" the beginning of the list
sc_blocklist = ["blocklist"] + sc_blocklist
sc_blocklist_path = pathlib.Path("../data/blocklist/sc_blocklist.txt").resolve()
sc_blocklist_path.parent.mkdir(parents=True, exist_ok=True)
with open(sc_blocklist_path, "w") as f:
    for item in sc_blocklist:
        f.write(f"{item}\n")


# In[8]:


sc_metadata_columns = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "image_set",
    "Well",
    "parent_organoid",
]
sc_features_columns = [
    col for col in sc_normalized.columns if col not in sc_metadata_columns
]
sc_features_df = sc_normalized.drop(columns=sc_metadata_columns, errors="ignore")


# In[9]:


# normalize the data
sc_fs_profiles = feature_select(
    sc_features_df,
    operation=feature_select_ops,
    features=sc_features_columns,
    blocklist_file=sc_blocklist_path,
)
sc_fs_profiles = pd.concat(
    [
        sc_normalized[sc_metadata_columns].reset_index(drop=True),
        sc_fs_profiles.reset_index(drop=True),
    ],
    axis=1,
)
sc_fs_profiles.to_parquet(sc_fs_output_path, index=False)
sc_fs_profiles.head()


# ### Normalize the organoid profiles

# In[10]:


organoid_normalized.head()


# In[11]:


organoid_blocklist = [
    x
    for x in organoid_normalized.columns
    if "Area" in x and ("MAX" in x or "MIN" in x or "BBOX" in x or "CENTER" in x)
]
# write the blocklist to a file
# add "blocklist" the beginning of the list
organoid_blocklist = ["blocklist"] + organoid_blocklist
organoid_blocklist_path = pathlib.Path(
    "../data/blocklist/organoid_blocklist.txt"
).resolve()
organoid_blocklist_path.parent.mkdir(parents=True, exist_ok=True)
with open(organoid_blocklist_path, "w") as f:
    for item in organoid_blocklist:
        f.write(f"{item}\n")


# In[12]:


organoid_metadata_columns = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "image_set",
    "Well",
    "single_cell_count",
]
organoid_features_columns = [
    col for col in organoid_normalized.columns if col not in organoid_metadata_columns
]
organoid_features_df = organoid_normalized.drop(
    columns=organoid_metadata_columns, errors="ignore"
)


# In[13]:


# normalize the data
organoid_fs_profiles = feature_select(
    organoid_features_df,
    operation=feature_select_ops,
    features=organoid_features_columns,
    blocklist_file=organoid_blocklist_path,
)
organoid_fs_profiles = pd.concat(
    [
        organoid_normalized[organoid_metadata_columns].reset_index(drop=True),
        organoid_fs_profiles.reset_index(drop=True),
    ],
    axis=1,
)
organoid_fs_profiles.to_parquet(organoid_fs_output_path, index=False)
organoid_fs_profiles.head()
