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


# ### Merge the sc and organoid profiles after aggregation
# 1. The single-cell parent organoid aggregated profile is merged with the fs organoid profile
# 2. The well level profiles are merged together
# 3. The consensus profiles are merged together
#

# In[3]:


# group the import paths by the type of aggregation
#######################################################################
# 1. The single-cell parent organoid aggregated profile is merged with the fs organoid profile
organoid_fs_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/3.organoid_fs_profiles.parquet"
).resolve(strict=True)
sc_agg_well_parent_organoid_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/4.sc_agg_well_parent_organoid_level_profiles.parquet"
).resolve(strict=True)

# ouput merged path
organoid_agg_well_parent_organoid_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/5.sc-organoid-sc_well_agg_parent_organoid_level_profiles.parquet"
).resolve()

########################################################################

# 2. The well level profiles are merged together
sc_agg_well_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/4.sc_agg_well_level_profiles.parquet"
).resolve(strict=True)

organoid_agg_well_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/4.organoid_agg_well_level_profiles.parquet"
).resolve(strict=True)

# output merged path
organoid_agg_well_merge_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/5.sc-organoid-sc_well_agg_merge_profiles.parquet"
).resolve()

###################################################################################

# 3. The consensus profiles are merged together

sc_consensus_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/4.sc_consensus_profiles.parquet"
).resolve(strict=True)

organoid_consensus_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/4.organoid_consensus_profiles.parquet"
).resolve(strict=True)

# output merged path
organoid_consensus_merge_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/5.sc-organoid_consensus_profiles.parquet"
).resolve()

###############################################################################


# In[4]:


organoid_fs = pd.read_parquet(organoid_fs_path)
sc_agg_well_parent_organoid = pd.read_parquet(sc_agg_well_parent_organoid_path)
sc_agg_well_parent_organoid_merge = sc_agg_well_parent_organoid.merge(
    organoid_fs,
    left_on=["Well", "parent_organoid"],
    right_on=["Well", "object_id"],
)

sc_agg_well_parent_organoid_merge.to_parquet(
    organoid_agg_well_parent_organoid_path, index=False
)
sc_agg_well_parent_organoid_merge.head()


# In[5]:


sc_agg_well = pd.read_parquet(sc_agg_well_path)
organoid_agg_well = pd.read_parquet(organoid_agg_well_path)
sc_agg_well_merge = sc_agg_well.merge(
    organoid_agg_well,
    on=["Well"],
)
sc_agg_well_merge.to_parquet(organoid_agg_well_merge_path, index=False)
sc_agg_well_merge.head()


# In[6]:


sc_consensus = pd.read_parquet(sc_consensus_path)
organoid_consensus = pd.read_parquet(organoid_consensus_path)
sc_consensus_merge = sc_consensus.merge(organoid_consensus, on=["treatment"])
sc_consensus_merge.to_parquet(organoid_consensus_merge_path, index=False)
sc_consensus_merge.head()
