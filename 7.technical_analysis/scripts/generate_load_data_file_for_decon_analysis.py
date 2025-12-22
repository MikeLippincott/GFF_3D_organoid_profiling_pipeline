#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import numpy as np
import pandas as pd
import psutil
import tifffile
import torch
import tqdm
from arg_parsing_utils import parse_args
from notebook_init_utils import (
    avoid_path_crash_bandicoot,
    bandicoot_check,
    init_notebook,
)
from torch import rand

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


if not in_notebook:
    arguments_dict = parse_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    channel = arguments_dict["channel"]

else:
    well_fov = "F3-1"
    patient = "NF0014_T1"

channels = ["405", "488", "555", "640"]


# In[3]:


decon_patients = [
    "NF0014_T1",
    "NF0014_T2",
    "NF0016_T1",
    "NF0018_T6",
    "NF0021_T1",
    "NF0030_T1",
    "NF0040_T1",
    "SARCO219_T2",
    "SARCO361_T1",
]
channels = ["405", "488", "555", "640"]


# In[4]:


decon_image_metric_path = pathlib.Path(
    "../results/decon_image_metrics/individual_files/"
).resolve()


# In[5]:


list_of_files_expected = []

for patient in decon_patients:
    for well_fov in os.listdir(f"{image_base_dir}/data/{patient}/zstack_images/"):
        for channel in channels:
            expected_file_path = pathlib.Path(
                f"{decon_image_metric_path}/{patient}_{well_fov}_{channel}_decon_image_metrics.parquet"
            ).resolve()
            list_of_files_expected.append(expected_file_path)

paths_present = decon_image_metric_path.glob("*_decon_image_metrics.parquet")
present_files = [path for path in paths_present]
files_to_run_or_rerun = set(list_of_files_expected) - set(present_files)
files_to_run_or_rerun = sorted(list(files_to_run_or_rerun))
files_to_run_or_rerun_df = pd.DataFrame(
    files_to_run_or_rerun, columns=["output_file_path"]
)
files_to_run_or_rerun_df["patient"] = files_to_run_or_rerun_df[
    "output_file_path"
].apply(lambda x: x.stem.split("_")[0] + "_" + x.stem.split("_")[1])
files_to_run_or_rerun_df["well_fov"] = files_to_run_or_rerun_df[
    "output_file_path"
].apply(lambda x: x.stem.split("_")[2].split("_")[0])

files_to_run_or_rerun_df.drop(columns=["output_file_path"], inplace=True)
files_to_run_or_rerun_df = files_to_run_or_rerun_df.drop_duplicates()


# In[6]:


# write the patient and wellfov to a load file
pathlib.Path("../loadfiles/").mkdir(parents=True, exist_ok=True)
with open("../loadfiles/decon_image_metrics_load_file.txt", "w") as f:
    for index, row in files_to_run_or_rerun_df.iterrows():
        f.write(f"{row['patient']}\t{row['well_fov']}\n")
