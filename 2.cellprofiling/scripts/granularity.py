#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import sys
import time

sys.path.append("../featurization_utils")

import cucim
import cupy as cp
import numpy
import numpy as np
import pandas as pd
import scipy
import skimage
from granularity_utils import measure_3D_granularity

# from granularity import measure_3D_granularity
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


image_set_path = pathlib.Path("../../data/NF0014/cellprofiler/C4-2/")


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


# In[ ]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[6]:


start_time = time.time()


# In[ ]:


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
            image=image_set_loader.image_set_dict[channel],
            label_image=image_set_loader.image_set_dict[compartment],
            channel_name=channel,
            compartment_name=compartment,
        )
        object_measurements = measure_3D_granularity(
            object_loader=object_loader,
            radius=10,  # radius of the sphere to use for granularity measurement
            granular_spectrum_length=2,  # usually 16 but 2 is used for testing for now
            subsample_size=0.25,  # subsample to 25% of the image to reduce computation time
            image_name=channel,
        )
        final_df = pd.DataFrame(object_measurements)
        # get the mean of each value in the array
        # melt the dataframe to wide format
        final_df = final_df.pivot_table(
            index=["object_id"], columns=["feature"], values=["value"]
        )
        final_df.columns = final_df.columns.droplevel()
        final_df = final_df.reset_index()
        # prepend compartment and channel to column names
        final_df.columns = [
            f"{compartment}_{channel}_{col}" for col in final_df.columns
        ]
        final_df["image_set"] = image_set_loader.image_set_name

        output_file = pathlib.Path(
            f"../results/{image_set_loader.image_set_name}/Granularity_{compartment}_{channel}_features.parquet"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_file)


# In[ ]:


print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
print("--- %s hours ---" % ((time.time() - start_time) / 3600))
