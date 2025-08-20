#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import sys

import duckdb
import pandas as pd

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
from segmentation_init_utils import parse_segmentation_args

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path("/home/lippincm/mnt/bandicoot").resolve(), root_dir
)


# In[2]:


log_directory = pathlib.Path(
    root_dir / "4.processing_image_based_profiles/logs/patients"
).resolve(strict=True)
# get a list of all files in the log directory
log_files = [f for f in log_directory.iterdir() if f.is_file() and f.suffix == ".log"]
total_runs = 0
passed_list = []
failed_list = []
for log_file in log_files:
    if "run_stats" in log_file.name:
        continue
    total_runs += 1
    with open(log_file, "r") as f:
        content = f.read()
        if "Traceback" in content:
            failed_list.append(log_file.name)
        else:
            passed_list.append(log_file.name)
print(f"Total runs: {total_runs}")
print(f"Passed runs: {len(passed_list)}")
print(f"Failed runs: {len(failed_list)}")
print(
    "Percentage of runs that passed: {:.2f}%".format(
        (len(passed_list) / total_runs) * 100
    )
)
