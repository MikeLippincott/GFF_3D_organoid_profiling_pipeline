#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib
import shutil
import sys

import numpy as np
import tqdm

sys.path.append(str(pathlib.Path("../../utils").resolve()))
from file_checking import check_number_of_files

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser(
        description="set up directories for the analysis of the data"
    )

    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="patient name, e.g. 'P01'",
    )

    args = argparser.parse_args()
    patient = args.patient
else:
    patient = "NF0014"

overwrite = True


# In[3]:


# set path to the processed data dir
processed_data_dir = pathlib.Path(f"../../data/{patient}/processed_data").resolve(
    strict=True
)
raw_input_dir = pathlib.Path(f"../../data/{patient}/zstack_images").resolve(strict=True)

cellprofiler_dir = pathlib.Path(f"../../data/{patient}/cellprofiler").resolve()
if cellprofiler_dir.exists():
    shutil.rmtree(cellprofiler_dir)
    cellprofiler_dir.mkdir(parents=True, exist_ok=True)
else:
    cellprofiler_dir.mkdir(parents=True, exist_ok=True)


# In[4]:


# perform checks for each directory
processed_data_dir_directories = list(processed_data_dir.glob("*"))
normalized_data_dir_directories = list(raw_input_dir.glob("*"))
cellprofiler_dir_directories = list(cellprofiler_dir.glob("*"))

print(
    f"""
      #################################################################################\n
      ## Checking the number of files in each subdirectory of:\n
      ## {processed_data_dir.absolute()}\n
      #################################################################################
      """
)
for file in processed_data_dir_directories:
    check_number_of_files(file, 10)


print(
    f"""
      #################################################################################\n
      ## Checking the number of files in each subdirectory of:\n
      ## {raw_input_dir.absolute()}\n
      #################################################################################
      """
)
for file in normalized_data_dir_directories:
    check_number_of_files(file, 5)


# ## Copy the normalized images to the cellprofiler images dir

# In[5]:


# get the list of dirs in the normalized_data_dir
norm_dirs = [x for x in raw_input_dir.iterdir() if x.is_dir()]
# copy each dir and files to cellprofiler_dir
for norm_dir in tqdm.tqdm(norm_dirs):
    dest_dir = pathlib.Path(cellprofiler_dir, norm_dir.name)
    if dest_dir.exists() and overwrite:
        shutil.rmtree(dest_dir)
        shutil.copytree(norm_dir, dest_dir)
    elif not dest_dir.exists():
        shutil.copytree(norm_dir, dest_dir)
    else:
        pass


# ## Copy files from processed dir to cellprofiler images dir

# In[6]:


# get a list of dirs in processed_data
dirs = [x for x in processed_data_dir.iterdir() if x.is_dir()]
file_extensions = {".tif", ".tiff"}
# get a list of files in each dir
for well_dir in tqdm.tqdm(dirs):
    files = [x for x in well_dir.iterdir() if x.is_file()]
    for file in files:
        if file.suffix in file_extensions:
            # copy each of the raw files to the cellprofiler_dir for feature extraction
            new_file_dir = pathlib.Path(
                cellprofiler_dir, well_dir.name, file.stem + file.suffix
            )
            shutil.copy(file, new_file_dir)


# In[7]:


jobs_to_rerun_path = pathlib.Path("../rerun_jobs.txt").resolve()
if jobs_to_rerun_path.exists():
    jobs_to_rerun_path.unlink()


# In[8]:


dirs_in_cellprofiler_dir = [x for x in cellprofiler_dir.iterdir() if x.is_dir()]
dirs_in_cellprofiler_dir = sorted(dirs_in_cellprofiler_dir)
for dir in tqdm.tqdm(dirs_in_cellprofiler_dir):
    if not check_number_of_files(dir, 12):
        with open(jobs_to_rerun_path, "a") as f:
            f.write(f"{dir.name}\n")


# In[9]:


# move an example to the example dir
example_dir = pathlib.Path("../animations/gif/C4-2").resolve(strict=True)
final_example_dir = pathlib.Path("../examples/segmentation_output/C4-2/gifs").resolve()
if final_example_dir.exists():
    shutil.rmtree(final_example_dir)


if example_dir.exists():
    shutil.copytree(example_dir, final_example_dir)
