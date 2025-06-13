#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile normalization.
# All profiles are normalized to the DMSO control treated profiles.

# In[1]:


import argparse
import pathlib

import pandas as pd
from pycytominer import normalize

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
sc_annotated_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/1.sc_annotated_profiles.parquet"
).resolve(strict=True)
organoid_annotated_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/1.organoid_annotated_profiles.parquet"
).resolve(strict=True)


# output path
sc_normalized_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/2.sc_normalized_profiles.parquet"
).resolve()
organoid_normalized_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/2.organoid_normalized_profiles.parquet"
).resolve()


# In[4]:


# read in the data
sc_annotated_profiles = pd.read_parquet(sc_annotated_path)
organoid_annotated_profiles = pd.read_parquet(organoid_annotated_path)


# ### Normalize the single-cell profiles

# In[5]:


sc_annotated_profiles.head()


# In[6]:


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
    col for col in sc_annotated_profiles.columns if col not in sc_metadata_columns
]


# In[7]:


# normalize the data
sc_normalized_profiles = normalize(
    sc_annotated_profiles,
    features=sc_features_columns,
    meta_features=sc_metadata_columns,
    method="standardize",
    samples="treatment == 'DMSO'",
)
sc_normalized_profiles.to_parquet(sc_normalized_output_path, index=False)
sc_normalized_profiles.head()


# ### Normalize the organoid profiles

# In[8]:


organoid_annotated_profiles.head()


# In[9]:


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
    col
    for col in organoid_annotated_profiles.columns
    if col not in organoid_metadata_columns
]
# normalize the data
organoid_normalized_profiles = normalize(
    organoid_annotated_profiles,
    features=organoid_features_columns,
    meta_features=organoid_metadata_columns,
    method="standardize",
    samples="treatment == 'DMSO'",
)
organoid_normalized_profiles.to_parquet(organoid_normalized_output_path, index=False)
organoid_normalized_profiles.head()
