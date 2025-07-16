#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
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
sys.path.append(f"{root_dir}/3.cellprofiling/featurization_utils/")
from loading_classes import ImageSetLoader

sys.path.append(str(pathlib.Path(f"{root_dir}/utils").resolve()))
from file_checking import check_number_of_files

# In[2]:


patient = "NF0014"
well_fov = "C2-1"
# set path to the processed data dir

image_set_path = pathlib.Path(
    f"{root_dir}/data/{patient}/profiling_input_images/{well_fov}/"  # just to get channels structure
)
patient_id_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
rerun_combinations_path = pathlib.Path(
    f"{root_dir}/3.cellprofiling/load_data/rerun_combinations.json"
).resolve()
patient_ids = pd.read_csv(
    patient_id_file_path, header=None, names=["patient_id"]
).patient_id.tolist()


# In[3]:


channel_mapping = {
    "DNA": "405",
    "AGP": "488",
    "ER": "555",
    "Mito": "640",
    "BF": "TRANS",
    "Nuclei": "nuclei_",
    "Cell": "cell_",
    "Cytoplasm": "cytoplasm_",
    "Organoid": "organoid_",
}
image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)

channels = image_set_loader.image_names
compartments = image_set_loader.compartments
channel_combinations = list(itertools.combinations(channels, 2))


# For each well fov there should be the following number of files:
# Of course this depends on if both CPU and GPU versions are run, but the CPU version is always run.
# | Feature Type | No. Compartments | No. Channels | No. Processors | Total No. Files |
# |--------------|------------------|---------------|----------------|-----------------|
# | AreaSizeShape | 4 | 1 | 2 | 8 |
# | Colocalization | 4 | 10 | 2 | 80 |
# | Granularity | 4 | 5 | 1 | 20 |
# | Intensity | 4 | 5 | 2 | 40 |
# | Neighbors | 1 | 1 | 1 | 1 |
# | Texture | 4 | 5 | 1 | 20 |
#
# Total no. files per well fov = 169
#

# In[4]:


feature_types = [
    "AreaSizeShape",
    "Colocalization",
    "Granularity",
    "Intensity",
    "Neighbors",
    "Texture",
]


# In[5]:


feature_list = []
# construct the file space

# area, size, shape
for compartment in compartments:
    for processor_type in ["CPU", "GPU"]:
        feature_list.append(f"AreaSizeShape_{compartment}_{processor_type}_features")
# colocalization
for channel in channel_combinations:
    for compartment in compartments:
        for processor_type in ["CPU", "GPU"]:
            feature_list.append(
                f"Colocalization_{compartment}_{channel[0]}.{channel[1]}_{processor_type}_features"
            )
# granularity
for channel in channels:
    for compartment in compartments:
        feature_list.append(f"Granularity_{compartment}_{channel}_CPU_features")
# intensity
for channel in channels:
    for compartment in compartments:
        for processor_type in ["CPU", "GPU"]:
            feature_list.append(
                f"Intensity_{compartment}_{channel}_{processor_type}_features"
            )
# neighbors
feature_list.append("Neighbors_Nuclei_DNA_CPU_features")
# texture
for channel in channels:
    for compartment in compartments:
        feature_list.append(f"Texture_{compartment}_{channel}_CPU_features")


# In[6]:


featurization_rerun_dict = {
    "patient": patient,
    "well_fov": well_fov,
    "feature": [],
    "compartment": [],
    "channel": [],
    "processor_type": [],
}


# In[7]:


for patient in patient_ids:
    featurization_data_dir = pathlib.Path(
        f"{root_dir}/data/{patient}/extracted_features/"
    ).resolve()

    # perform checks for each directory
    featurization_data_dirs = list(featurization_data_dir.glob("*"))
    featurization_data_dirs = [d for d in featurization_data_dirs if d.is_dir()]

    for dir in featurization_data_dirs:
        if dir.name != "run_stats":
            if not check_number_of_files(dir, 169):
                # find the missing files
                # cross reference the files in the directory
                # with the expected feature list
                existing_files = set(f.name for f in dir.glob("*"))
                existing_files = [f.stem for f in dir.glob("*") if f.is_file()]
                missing_files = set(feature_list) - set(existing_files)
                assert len(missing_files) >= 0, "There should be no missing files"
                assert len(missing_files) <= 169, (
                    "There should be at most 169 missing files"
                )
                print(
                    len(missing_files) + len(existing_files), "files in the directory"
                )
                assert len(missing_files) + len(existing_files) == 169, (
                    "There should be exactly 169 files in the directory"
                )
                if missing_files:
                    for missing_file in missing_files:
                        featurization_rerun_dict["patient"] = patient
                        featurization_rerun_dict["well_fov"] = well_fov
                        featurization_rerun_dict["feature"].append(
                            missing_file.split("_")[0]
                        )
                        featurization_rerun_dict["compartment"].append(
                            missing_file.split("_")[1]
                        )
                        if missing_file.split("_")[0] == "Colocalization":
                            featurization_rerun_dict["channel"].append(
                                missing_file.split("_")[2].split(".")[0]
                                + "."
                                + missing_file.split("_")[2].split(".")[1]
                            )

                        else:
                            featurization_rerun_dict["channel"].append(
                                missing_file.split("_")[2]
                            )
                        featurization_rerun_dict["processor_type"].append(
                            missing_file.split("_")[3]
                        )

print(f"Total number of files expected: {len(featurization_data_dirs) * 169}")
print(
    f"Total number of files found: {sum([len(list(d.glob('*'))) for d in featurization_data_dirs])}"
)


# In[8]:


df = pd.DataFrame(featurization_rerun_dict)
df.to_json(rerun_combinations_path, orient="records", indent=4)
df.head()


# In[9]:


df
