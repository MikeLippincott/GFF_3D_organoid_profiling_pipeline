#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import shutil
import sys

import pandas as pd
import synapseclient
import synapseutils

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

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm
profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


profiles_dir = pathlib.Path(f"{profile_base_dir}/data/all_patient_profiles").resolve()
# get all patient profile dirs
profile_dirs = [
    d
    for d in profiles_dir.iterdir()
    if ".parquet" in str(d) and "featurization" not in str(d)
]


# In[ ]:


sage_profiles_dir = pathlib.Path(
    "../data_for_sage/Raw Data/bulk quantification/"
    # see comments below if the spaces in the path annoy you...
    # to match the expected input dir for sage
    # note, the data_for_sage part of the dir does not get synced to synapse
    # this path provided syncs everything that matches ( or not ) a pattern
    # on synapse
    # so we need to make sure the directory structure is correct
    # also, this directory should be temporary and not checked into git
    # so it is in the .gitignore file just in case
    # but is also deleted at the end of this notebook
).resolve()
if sage_profiles_dir.exists():
    shutil.rmtree(sage_profiles_dir)
sage_profiles_dir.mkdir(parents=True, exist_ok=True)


# In[ ]:


# get each of the profiles and split them by:
# patient tumor, treatment, dose+units
for profile_file_path in tqdm.tqdm(profile_dirs):
    profile_name = profile_file_path.stem.split("_profiles")[0]
    profile_name = profile_name.replace("fs", "feature_selected")
    profile_name = profile_name.replace("agg", "aggregated")
    df = pd.read_parquet(profile_file_path)
    df["Metadata_dose_plus_units"] = (
        df["Metadata_dose"].astype(str) + "_" + df["Metadata_unit"]
    )
    df.to_parquet(
        f"{sage_profiles_dir}/{profile_name}.parquet",
        partition_cols=[
            "Metadata_patient_tumor",
            "Metadata_treatment",
            "Metadata_dose_plus_units",
        ],
    )


# In[5]:


# get a list of all output files and dirs
output_files = list(sage_profiles_dir.glob("**/*"))
output_dirs = [d for d in output_files if d.is_dir()]
# rename the most nested dirs first to avoid issues with parent dirs being renamed before child dirs
_ = [
    d.rename(d.parent / d.name.replace("=", "_"))
    for d in sorted(output_dirs, key=lambda x: len(x.parts), reverse=True)
    if "=" in d.name
]
output_files = list(sage_profiles_dir.glob("**/*"))
output_dirs = [d for d in output_files if d.is_dir()]
_ = [
    d.rename(d.parent / d.name.replace("%", "percent_"))
    for d in sorted(output_dirs, key=lambda x: len(x.parts), reverse=True)
    if "%" in d.name
]
output_files = list(sage_profiles_dir.glob("**/*"))
output_dirs = [d for d in output_files if d.is_dir()]
_ = [
    shutil.rmtree(d)
    for d in output_dirs
    if "Metadata_treatment___HIVE_DEFAULT_PARTITION__" in d.name
]


# In[6]:


README_path = pathlib.Path("../README.md").resolve()
sage_readme_path = pathlib.Path(f"{sage_profiles_dir}/README.md").resolve()
shutil.copy(README_path, sage_readme_path)


# ## Upload the processed profiles to Synapse for Sage processing

# Tutorial on how to use synapse client: https://python-docs.synapse.org/en/stable/tutorials/python/upload_data_in_bulk/

# In[7]:


# note, must run synapse config first in terminal to set up .synapseConfig file
# or set some environment variables
syn = synapseclient.login()


# In[ ]:


my_project_id = my_project_id = syn.findEntityId(
    name="A deep learning microscopy framework for NF1 patient-derived organoid drug screening"
)
DIRECTORY_FOR_MY_PROJECT = os.path.join(
    "..", "data_for_sage/"
)  # tried using pathlib and it throws an error in the generate sync manifest function
PATH_TO_MANIFEST_FILE = os.path.join(".", "manifest-for-upload.tsv")


# In[9]:


# generate the manifest file to sync on
synapseutils.generate_sync_manifest(
    syn=syn,
    directory_path=DIRECTORY_FOR_MY_PROJECT,
    parent_id=my_project_id,
    manifest_path=PATH_TO_MANIFEST_FILE,
)


# In[10]:


# sync the files to synapse
synapseutils.syncToSynapse(
    syn=syn, manifestFile=PATH_TO_MANIFEST_FILE, sendMessages=False
)
