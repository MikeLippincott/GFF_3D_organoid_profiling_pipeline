#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import sys
import time

import psutil

sys.path.append("../featurization_utils")
import itertools
from itertools import product

import pandas as pd
from colocalization_utils import (
    measure_3D_colocalization,
    prepare_two_images_for_colocalization,
)
from colocalization_utils_gpu import (
    measure_3D_colocalization_gpu,
    prepare_two_images_for_colocalization_gpu,
)
from featurization_parsable_arguments import parse_featurization_args_colocalization
from loading_classes import ImageSetLoader, TwoObjectLoader
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

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[2]:


if not in_notebook:
    arguments_dict = parse_featurization_args_colocalization()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    channel1 = arguments_dict["channel1"]
    channel2 = arguments_dict["channel2"]
    compartment = arguments_dict["compartment"]
    processor_type = arguments_dict["processor_type"]

else:
    well_fov = "C4-2"
    patient = "NF0014"
    channel1 = "DNA"
    channel2 = "ER"
    compartment = "Nuclei"
    processor_type = "GPU"

image_set_path = pathlib.Path(f"../../data/{patient}/cellprofiler/{well_fov}/")
output_parent_path = pathlib.Path(
    f"../../data/{patient}/extracted_features/{well_fov}/"
)
output_parent_path.mkdir(parents=True, exist_ok=True)


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


# In[4]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[5]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[6]:


coloc_loader = TwoObjectLoader(
    image_set_loader=image_set_loader,
    compartment=compartment,
    channel1=channel1,
    channel2=channel2,
)

output_dir = pathlib.Path(
    output_parent_path
    / f"Colocalization_{compartment}_{channel1}.{channel2}_features.parquet"
)
list_of_dfs = []
for object_id in coloc_loader.object_ids:
    if processor_type == "CPU":
        cropped_image1, cropped_image2 = prepare_two_images_for_colocalization(
            label_object1=coloc_loader.label_image,
            label_object2=coloc_loader.label_image,
            image_object1=coloc_loader.image1,
            image_object2=coloc_loader.image2,
            object_id1=object_id,
            object_id2=object_id,
        )
        colocalization_features = measure_3D_colocalization(
            cropped_image_1=cropped_image1,
            cropped_image_2=cropped_image2,
            thr=15,
            fast_costes="Accurate",
        )
    elif processor_type == "GPU":
        cropped_image1, cropped_image2 = prepare_two_images_for_colocalization_gpu(
            label_object1=coloc_loader.label_image,
            label_object2=coloc_loader.label_image,
            image_object1=coloc_loader.image1,
            image_object2=coloc_loader.image2,
            object_id1=object_id,
            object_id2=object_id,
        )
        colocalization_features = measure_3D_colocalization_gpu(
            cropped_image_1=cropped_image1,
            cropped_image_2=cropped_image2,
            thr=15,
            fast_costes="Accurate",
        )
    else:
        raise ValueError(
            f"Processor type {processor_type} is not supported. Use 'CPU' or 'GPU'."
        )
    coloc_df = pd.DataFrame(colocalization_features, index=[0])
    coloc_df.columns = [
        f"Colocalization_{compartment}_{channel1}.{channel2}_{col}"
        for col in coloc_df.columns
    ]
    coloc_df.insert(0, "object_id", object_id)
    coloc_df.insert(1, "image_set", image_set_loader.image_set_name)
    list_of_dfs.append(coloc_df)
coloc_df = pd.concat(list_of_dfs, ignore_index=True)
coloc_df.to_parquet(output_dir)


# In[7]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    feature_type="Colocalization",
    well_fov=well_fov,
    patient_id=patient,
    channel=f"{channel1}.{channel2}",
    compartment=compartment,
    CPU_GPU=processor_type,
    output_file_dir=pathlib.Path(
        f"../../data/{patient}/extracted_features/run_stats/{well_fov}_Colocalization_{channel1}.{channel2}_{compartment}_{processor_type}.parquet"
    ),
)
