#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import pathlib
import sys
import time

import psutil

sys.path.append("../featurization_utils")
import numpy as np
import pandas as pd
import scipy
import skimage
from loading_classes import ImageSetLoader, ObjectLoader
from neighbors_utils import measure_3D_number_of_neighbors
from resource_profiling_util import get_mem_and_time_profiling

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--well_fov",
        type=str,
        default="None",
        help="Well and field of view to process, e.g. 'A01_1'",
    )
    argparser.add_argument(
        "--patient",
        type=str,
        help="Patient ID, e.g. 'NF0014'",
    )

    args = argparser.parse_args()
    well_fov = args.well_fov
    patient = args.patient
    if well_fov == "None":
        raise ValueError(
            "Please provide a well and field of view to process, e.g. 'A01_1'"
        )

else:
    well_fov = "C4-2"
    patient = "NF0014"

image_set_path = pathlib.Path(f"../../data/{patient}/cellprofiler/{well_fov}/")
output_parent_path = pathlib.Path(
    f"../../data/{patient}/extracted_features/{well_fov}/"
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


# loop through each compartment and channel
# and measure the number of neighbors
# for each compartment

compartment = "Nuclei"
channel = "DNA"
object_loader = ObjectLoader(
    image_set_loader.image_set_dict[channel],
    image_set_loader.image_set_dict[compartment],
    channel,
    compartment,
)
neighbors_out_dict = measure_3D_number_of_neighbors(
    object_loader=object_loader,
    distance_threshold=10,
    anisotropy_factor=image_set_loader.anisotropy_factor,
)
final_df = pd.DataFrame(neighbors_out_dict)
final_df.insert(0, "image_set", image_set_loader.image_set_name)

output_file = pathlib.Path(
    output_parent_path / f"Neighbors_{compartment}_{channel}_features.parquet"
)
output_file.parent.mkdir(parents=True, exist_ok=True)
final_df.to_parquet(output_file)
final_df.head()


# In[7]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    feature_type="Neighbors",
    well_fov=well_fov,
    CPU_GPU="CPU",
)
