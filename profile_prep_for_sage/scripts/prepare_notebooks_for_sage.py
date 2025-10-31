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


def recursive_remove_empty_dirs(path: pathlib.Path):
    """
    Recursively remove empty directories.

    Parameters
    ----------
    path : pathlib.Path
        The root directory to start removing empty directories from.
    """
    if not path.is_dir():
        return
    for child in path.iterdir():
        recursive_remove_empty_dirs(child)
    if not any(path.iterdir()):
        path.rmdir()


# In[3]:


profiles_dir = pathlib.Path(f"{profile_base_dir}/data/all_patient_profiles").resolve()
# get all patient profile dirs
profile_dirs = [
    d
    for d in profiles_dir.rglob("*.parquet")
    if "featurization" not in str(d)
]


# In[4]:


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


# In[5]:


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


# In[6]:


output_dirs = [d for d in list(sage_profiles_dir.glob("**/*")) if d.is_dir()]


# In[7]:


# get a list of all output files and dirs
output_dirs = sorted(
    [d for d in list(sage_profiles_dir.glob("**/*")) if d.is_dir()],
    key=lambda x: len(x.parts),
    reverse=True,
)
# get a list of all output files and dirs
output_dirs = [d for d in list(sage_profiles_dir.glob("**/*")) if d.is_dir()]

# rename the most nested dirs first to avoid issues with parent dirs being renamed before child dirs
_ = [
    d.rename(d.parent / d.name.replace("=", "_"))
    for d in sorted(output_dirs, key=lambda x: len(x.parts), reverse=True)
    if "=" in d.name
]
output_dirs = [d for d in list(sage_profiles_dir.glob("**/*")) if d.is_dir()]
_ = [
    d.rename(d.parent / d.name.replace("%", "percent"))
    for d in sorted(output_dirs, key=lambda x: len(x.parts), reverse=True)
    if "%" in d.name
]
output_dirs = [d for d in list(sage_profiles_dir.glob("**/*")) if d.is_dir()]
_ = [
    shutil.rmtree(d)
    for d in output_dirs
    if "Metadata_treatment___HIVE_DEFAULT_PARTITION__" in d.name
]
output_dirs = [d for d in list(sage_profiles_dir.glob("**/*")) if d.is_dir()]

# replace Metadata_patient_tumor_ with ""
_ = [
    d.rename(d.parent / d.name.replace("Metadata_patient_tumor_", ""))
    for d in output_dirs
    if "Metadata_patient_tumor_" in d.name
]
output_dirs = [d for d in list(sage_profiles_dir.glob("**/*")) if d.is_dir()]

# replace Metadata_treatment_ with ""
_ = [
    d.rename(d.parent / d.name.replace("Metadata_treatment_", ""))
    for d in output_dirs
    if "Metadata_treatment_" in d.name
]
output_dirs = [d for d in list(sage_profiles_dir.glob("**/*")) if d.is_dir()]

# replace Metadata_dose_plus_units_ with ""
_ = [
    d.rename(d.parent / d.name.replace("Metadata_dose_plus_units_", ""))
    for d in output_dirs
    if "Metadata_dose_plus_units_" in d.name
]


# In[8]:


output_files = [f for f in sage_profiles_dir.glob("**/*") if f.is_file()]
# loop through and rename files to contain the proper metadata
for file in output_files:
    parent_dir = str(file).split(".parquet/")[0]
    new_file_name = (
        str(file)
        .split(".parquet/")[1]
        .replace("/", "_")
        .replace(f"{str(file.stem)}.", "")
    )
    new_file_path = pathlib.Path(parent_dir) / new_file_name
    new_file_path.parent.mkdir(parents=True, exist_ok=True)
    file.rename(new_file_path)


# In[9]:


recursive_remove_empty_dirs(sage_profiles_dir)


# In[10]:


README_path = pathlib.Path("../README.md").resolve()
sage_readme_path = pathlib.Path(f"{sage_profiles_dir}/README.md").resolve()
shutil.copy(README_path, sage_readme_path)


# ## Upload the processed profiles to Synapse for Sage processing

# Tutorial on how to use synapse client: https://python-docs.synapse.org/en/stable/tutorials/python/upload_data_in_bulk/

# In[11]:


# note, must run synapse config first in terminal to set up .synapseConfig file
# or set some environment variables
syn = synapseclient.login()


# In[12]:


my_project_id = my_project_id = syn.findEntityId(
    name="A deep learning microscopy framework for NF1 patient-derived organoid drug screening"
)
DIRECTORY_FOR_MY_PROJECT = os.path.join(
    "..", "data_for_sage/"
)  # tried using pathlib and it throws an error in the generate sync manifest function
PATH_TO_MANIFEST_FILE = os.path.join(".", "manifest-for-upload.tsv")


# In[13]:


# generate the manifest file to sync on
synapseutils.generate_sync_manifest(
    syn=syn,
    directory_path=DIRECTORY_FOR_MY_PROJECT,
    parent_id=my_project_id,
    manifest_path=PATH_TO_MANIFEST_FILE,
)


# In[14]:


# sync the files to synapse
synapseutils.syncToSynapse(
    syn=syn, manifestFile=PATH_TO_MANIFEST_FILE, sendMessages=False
)
