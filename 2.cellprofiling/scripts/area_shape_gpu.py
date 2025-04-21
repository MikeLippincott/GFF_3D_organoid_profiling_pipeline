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
import skimage
from area_size_shape_utils_gpu import measure_3D_area_size_shape_gpu
from loading_classes import ImageSetLoader, ObjectLoader

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

    args = argparser.parse_args()
    well_fov = args.well_fov
    if well_fov == "None":
        raise ValueError(
            "Please provide a well and field of view to process, e.g. 'A01_1'"
        )

    image_set_path = pathlib.Path(f"../../data/NF0014/cellprofiler/{well_fov}/")
else:
    well_fov = "C4-2"
    image_set_path = pathlib.Path("../../data/NF0014/cellprofiler/C4-2/")


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


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_n_compartment_mapping,
)


# In[5]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[6]:


for compartment in tqdm(
    image_set_loader.compartments, desc="Processing compartments", position=0
):
    for channel in tqdm(
        image_set_loader.image_names,
        desc="Processing channels",
        leave=False,
        position=1,
    ):
        object_loader = ObjectLoader(
            image_set_loader.image_set_dict[channel],
            image_set_loader.image_set_dict[compartment],
            channel,
            compartment,
        )

        # area, size, shape
        size_shape_dict = measure_3D_area_size_shape_gpu(
            image_set_loader=image_set_loader,
            object_loader=object_loader,
        )
        final_df = pd.DataFrame(size_shape_dict)

        # prepend compartment and channel to column names
        for col in final_df.columns:
            if col not in ["object_id"]:
                final_df.rename(
                    columns={col: f"Area.Size.Shape_{compartment}_{channel}_{col}"},
                    inplace=True,
                )
        final_df.insert(1, "image_set", image_set_loader.image_set_name)

        output_file = pathlib.Path(
            f"../results/{image_set_loader.image_set_name}/AreaSize_Shape_{compartment}_{channel}_features.parquet"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_file)
        final_df.head()


# In[7]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
print(f"Memory usage: {end_mem - start_mem:.2f} MB")
print("Texture time:")
print("--- %s seconds ---" % (end_time - start_time))
print("--- %s minutes ---" % ((end_time - start_time) / 60))
print("--- %s hours ---" % ((end_time - start_time) / 3600))
# make a df of the run stats
run_stats = pd.DataFrame(
    {
        "start_time": [start_time],
        "end_time": [end_time],
        "start_mem": [start_mem],
        "end_mem": [end_mem],
        "time_taken": [(end_time - start_time)],
        "mem_usage": [(end_mem - start_mem)],
        "gpu": [True],
        "well_fov": [well_fov],
        "feature_type": ["area_size_shape"],
    }
)
# save the run stats to a file
run_stats_file = pathlib.Path(
    f"../results/run_stats/{well_fov}_area_size_shape_gpu.parquet"
)
run_stats_file.parent.mkdir(parents=True, exist_ok=True)
run_stats.to_parquet(run_stats_file)
