#!/usr/bin/env python
# coding: utf-8

# This notebook combines all well fovs for each patient into a single file.
#

# In[1]:


import argparse
import pathlib

import duckdb
import pandas as pd

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


# set paths
profiles_path = pathlib.Path(f"../../data/{patient}/image_based_profiles/").resolve(
    strict=True
)
# output_paths
sc_merged_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/0.sc_merged_profiles.parquet"
).resolve()
organoid_merged_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/0.organoid_merged_profiles.parquet"
).resolve()


# In[4]:


# get all profiles in the directory recursively
profiles = list(profiles_path.glob("**/*.parquet"))
# filter out profiles that are not related
profiles = [x for x in profiles if "related" in str(x)]


# In[5]:


sc_profiles = [str(x) for x in profiles if "sc" in str(x.name)]
organoid_profiles = [str(x) for x in profiles if "organoid" in str(x.name)]


# In[6]:


# concat all sc profiles with duckdb
with duckdb.connect() as conn:
    sc_profile = conn.execute(f"SELECT * FROM read_parquet({sc_profiles})").df()
    organoid_profile = conn.execute(
        f"SELECT * FROM read_parquet({organoid_profiles})"
    ).df()
print(f"Single-cell profiles concatenated. Shape: {sc_profile.shape}")
print(f"Organoid profiles concatenated. Shape: {organoid_profile.shape}")


# In[7]:


sc_profile.to_parquet(sc_merged_output_path, index=False)
organoid_profile.to_parquet(organoid_merged_output_path, index=False)
print(f"Single-cell profiles saved to {sc_merged_output_path}")
print(f"Organoid profiles saved to {organoid_merged_output_path}")
