#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile aggregation.

# In[1]:


import argparse
import pathlib

import pandas as pd
from pycytominer import aggregate

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
sc_fs_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/3.sc_fs_profiles.parquet"
).resolve(strict=True)
organoid_fs_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/3.organoid_fs_profiles.parquet"
).resolve(strict=True)


# output path
sc_agg_well_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/4.sc_agg_well_level_profiles.parquet"
).resolve()
sc_agg_well_parent_organoid_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/4.sc_agg_well_parent_organoid_level_profiles.parquet"
).resolve()
sc_consensus_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/4.sc_consensus_profiles.parquet"
).resolve()

organoid_agg_well_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/4.organoid_agg_well_level_profiles.parquet"
).resolve()
organoid_consensus_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/4.organoid_consensus_profiles.parquet"
).resolve()


# In[4]:


# read in the data
sc_fs = pd.read_parquet(sc_fs_path)
organoid_fs = pd.read_parquet(organoid_fs_path)


# ### Aggregate the single cell profiles
# We will aggregated with a few different stratifications:
# 1. Well
# 2. Well and parent organoid
# 3. Treatment - i.e. the consensus profile for each treatment

# In[5]:


sc_fs.head()


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
sc_features_columns = [col for col in sc_fs.columns if col not in sc_metadata_columns]
sc_features_df = sc_fs.drop(columns=sc_metadata_columns, errors="ignore")


# In[7]:


# stratification approach #1
sc_well_agg = aggregate(
    population_df=sc_fs,
    strata=["Well"],
    features=sc_features_columns,
    operation="median",
)
sc_well_agg.to_parquet(sc_agg_well_output_path, index=False)

# stratification approach #2
sc_well_parent_organoid_agg = aggregate(
    population_df=sc_fs,
    strata=["Well", "parent_organoid"],
    features=sc_features_columns,
    operation="median",
)
sc_well_parent_organoid_agg.to_parquet(
    sc_agg_well_parent_organoid_output_path, index=False
)
# stratification approach #3
sc_consensus = aggregate(  # a.k.a. consensus
    population_df=sc_fs,
    strata=["treatment"],
    features=sc_features_columns,
    operation="median",
)
sc_consensus.to_parquet(sc_consensus_output_path, index=False)


# ### Aggregate the organoid profiles
# We will aggregated with a few different stratifications:
# 1. Well
# 2. Treatment - i.e. the consensus profile for each treatment

# In[8]:


organoid_fs.head()


# In[9]:


organoid_metadata_columns = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "image_set",
    "Well",
    "parent_organoid",
]
organoidfeatures_columns = [
    col for col in organoid_fs.columns if col not in organoid_metadata_columns
]
organoid_features_df = organoid_fs.drop(columns=sc_metadata_columns, errors="ignore")


# In[10]:


# stratification approach #1
organoid_well_agg = aggregate(
    population_df=organoid_fs,
    strata=["Well"],
    features=organoidfeatures_columns,
    operation="median",
)
organoid_well_agg.to_parquet(organoid_agg_well_output_path, index=False)

# stratification approach #2
organoid_consensus = aggregate(  # a.k.a. consensus
    population_df=organoid_fs,
    strata=["treatment"],
    features=organoidfeatures_columns,
    operation="median",
)
organoid_consensus.to_parquet(organoid_consensus_output_path, index=False)
