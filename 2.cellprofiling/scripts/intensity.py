#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys
import time

sys.path.append("../featurization_utils")
import numpy as np
import pandas as pd
import scipy
import skimage
from intensity_utils import measure_3D_intensity_CPU
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
            image_set_loader.image_set_dict[channel],
            image_set_loader.image_set_dict[compartment],
            channel,
            compartment,
        )
        output_dict = measure_3D_intensity_CPU(object_loader)
        final_df = pd.DataFrame(output_dict)
        # prepend compartment and channel to column names
        final_df.columns = [
            f"{compartment}_{channel}_{col}" for col in final_df.columns
        ]
        final_df["image_set"] = image_set_loader.image_set_name
        final_df["feature"] = (
            "Intensity_"
            + final_df[f"{compartment}_{channel}_compartment"]
            + "_"
            + final_df[f"{compartment}_{channel}_channel"]
            + "_"
            + final_df[f"{compartment}_{channel}_feature_name"]
        )
        final_df.rename(
            columns={f"{compartment}_{channel}_object_id": "objectID"}, inplace=True
        )
        final_df.drop(
            columns=[
                f"{compartment}_{channel}_compartment",
                f"{compartment}_{channel}_channel",
                f"{compartment}_{channel}_feature_name",
            ],
            inplace=True,
        )
        # pivot wide
        final_df = final_df.pivot(
            index=["objectID", "image_set"],
            columns="feature",
            values=f"{compartment}_{channel}_value",
        )
        final_df.reset_index(inplace=True)

        output_file = pathlib.Path(
            f"../results/{image_set_loader.image_set_name}/Intensity_{compartment}_{channel}_features.parquet"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_file)


# In[7]:


print("Intensity time:")
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
print("--- %s hours ---" % ((time.time() - start_time) / 3600))
