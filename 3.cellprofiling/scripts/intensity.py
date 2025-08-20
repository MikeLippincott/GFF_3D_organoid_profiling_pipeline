#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import sys
import time
from itertools import product

import pandas as pd
import psutil

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

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
from featurization_parsable_arguments import parse_featurization_args
from intensity_utils import measure_3D_intensity_CPU, measure_3D_intensity_gpu
from loading_classes import ImageSetLoader, ObjectLoader
from resource_profiling_util import get_mem_and_time_profiling

# In[ ]:


if not in_notebook:
    arguments_dict = parse_featurization_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    channel = arguments_dict["channel"]
    compartment = arguments_dict["compartment"]
    processor_type = arguments_dict["processor_type"]


else:
    well_fov = "G4-6"
    patient = "NF0014_T1"
    channel = "AGP"
    compartment = "Nuclei"
    processor_type = "CPU"

image_set_path = pathlib.Path(
    f"{root_dir}/data/{patient}/profiling_input_images/{well_fov}/"
)
output_parent_path = pathlib.Path(
    f"{root_dir}/data/{patient}/extracted_features/{well_fov}/"
)
output_parent_path.mkdir(parents=True, exist_ok=True)


# In[3]:


channel_n_compartment_mapping = {
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


# In[4]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[5]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_n_compartment_mapping,
)


# In[6]:


object_loader = ObjectLoader(
    image_set_loader.image_set_dict[channel],
    image_set_loader.image_set_dict[compartment],
    channel,
    compartment,
)
if processor_type == "GPU":
    output_dict = measure_3D_intensity_gpu(object_loader)
elif processor_type == "CPU":
    output_dict = measure_3D_intensity_CPU(object_loader)
else:
    raise ValueError(
        f"Processor type {processor_type} is not supported. Use 'CPU' or 'GPU'."
    )
final_df = pd.DataFrame(output_dict)
# prepend compartment and channel to column names
final_df = final_df.pivot(
    index=["object_id"],
    columns="feature_name",
    values="value",
).reset_index()
for col in final_df.columns:
    if col == "object_id":
        continue
    else:
        final_df.rename(
            columns={col: f"Intensity_{compartment}_{channel}_{col}"},
            inplace=True,
        )

final_df.insert(0, "image_set", image_set_loader.image_set_name)

output_file = pathlib.Path(
    output_parent_path
    / f"Intensity_{compartment}_{channel}_{processor_type}_features.parquet"
)
output_file.parent.mkdir(parents=True, exist_ok=True)
final_df.to_parquet(output_file)


# In[7]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    feature_type="Intensity",
    well_fov=well_fov,
    patient_id=patient,
    channel=channel,
    compartment=compartment,
    CPU_GPU=processor_type,
    output_file_dir=pathlib.Path(
        f"{root_dir}/data/{patient}/extracted_features/run_stats/{well_fov}_{channel}_{compartment}_Intensity_{processor_type}.parquet"
    ),
)
