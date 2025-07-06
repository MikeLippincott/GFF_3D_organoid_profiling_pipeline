#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

import numpy as np
import pandas as pd
import tqdm

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
    # check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")

sys.path.append(str(pathlib.Path(f"{root_dir}/utils").resolve()))
from file_checking import check_number_of_files

# In[2]:


patientIDS_file = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True)
if not patientIDS_file.is_file():
    raise FileNotFoundError(f"File {patientIDS_file} not found.")
patientIDS = []
with open(patientIDS_file, "r") as f:
    for line in f:
        patientIDS.append(line.strip())
patientIDS


# In[3]:


# set the correct number of files to check for each directory
n_files = {
    "segmentation_data": 16,
    "zstack_data": 5,
    "profiling_input_images": 9,
}


# In[4]:


rerun_dict = {
    "patient": [],
    "well_fov": [],
    "zstack_counts": [],
    "segmentation_counts": [],
    "profiling_input_images_counts": [],
}
for patient in patientIDS:
    if not patient.isalnum():
        raise ValueError(f"Patient ID {patient} is not alphanumeric.")

    # set path to the processed data dir
    segmentation_data_dir = pathlib.Path(
        f"{root_dir}/data/{patient}/segmentation_masks/"
    ).resolve()
    zstack_dir = pathlib.Path(f"{root_dir}/data/{patient}/zstack_images/").resolve(
        strict=True
    )
    profiling_input_images_dir = pathlib.Path(
        f"{root_dir}/data/{patient}/profiling_input_images/"
    ).resolve()
    profiling_input_images_dir.mkdir(parents=True, exist_ok=True)
    well_fovs = [d.name for d in zstack_dir.glob("*") if d.is_dir()]

    for well_fov in well_fovs:
        rerun_dict["patient"].append(patient)
        rerun_dict["well_fov"].append(well_fov)
        if (
            check_number_of_files(zstack_dir / well_fov, n_files["zstack_data"])
            is not None
        ):
            rerun_dict["zstack_counts"].append(
                check_number_of_files(zstack_dir / well_fov, n_files["zstack_data"])[1]
            )
        else:
            rerun_dict["zstack_counts"].append(n_files["zstack_data"])
        if (
            check_number_of_files(
                segmentation_data_dir / well_fov, n_files["segmentation_data"]
            )
            is not None
        ):
            rerun_dict["segmentation_counts"].append(
                check_number_of_files(
                    segmentation_data_dir / well_fov, n_files["segmentation_data"]
                )[1]
            )
        else:
            rerun_dict["segmentation_counts"].append(n_files["segmentation_data"])
        if (
            check_number_of_files(
                profiling_input_images_dir / well_fov, n_files["profiling_input_images"]
            )
            is not None
        ):
            rerun_dict["profiling_input_images_counts"].append(
                check_number_of_files(
                    profiling_input_images_dir / well_fov,
                    n_files["profiling_input_images"],
                )[1]
            )
        else:
            rerun_dict["profiling_input_images_counts"].append(
                n_files["profiling_input_images"]
            )


# In[5]:


rerun_df = pd.DataFrame(rerun_dict)
rerun_df["rerun_boolean"] = np.where(
    (rerun_df["segmentation_counts"] != n_files["segmentation_data"])
    | (rerun_df["zstack_counts"] != n_files["zstack_data"])
    | (rerun_df["profiling_input_images_counts"] != n_files["profiling_input_images"]),
    True,
    False,
)
rerun_df.head()


# In[6]:


# write the patient and well_fov to a file to be pared by a shell script
# This will be used to rerun the segmentation and zstack processing
rerun_file = pathlib.Path(f"{root_dir}/2.segment_images/rerun_jobs.txt").resolve()
with open(rerun_file, "w") as f:
    for index, row in rerun_df.iterrows():
        if row["rerun_boolean"]:
            f.write(f"{row['patient']}\t{row['well_fov']}\n")


# In[7]:


print(f"""
For {len((rerun_df["patient"].unique()))} patients,
{len(rerun_df)} wells/fovs were checked,
{len(rerun_df.loc[rerun_df["rerun_boolean"]])} wells/fovs need to be rerun.
This is determined by the number of files in the segmentation,
zstack and profiling input images directories.
""")
