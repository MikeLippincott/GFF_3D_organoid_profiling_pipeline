#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys
import time

sys.path.append("../featurization")

import cucim
import cucim.skimage.morphology
import cupy as cp
import cupyx.scipy.ndimage
import numpy
import numpy as np
import pandas as pd
import scipy
import skimage
import tqdm
from granularity_utils import measure_3D_granularity_gpu
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


# In[4]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[5]:


def granularity_feature(length):
    C_GRANULARITY = "GRANULARITY.%s"
    return C_GRANULARITY % (length)


class ObjectRecord:
    def __init__(self, object_loader, object_index):
        self.object_index = object_index
        self.labels = object_loader.objects.copy()
        # select the object
        self.labels[self.labels != object_index] = 0
        self.image = object_loader.image.copy()
        self.image[self.labels != object_index] = 0

        # self.labels[self.labels == object_index] = 1
        # self.labels[~object_index] = 0

        self.nobjects = len(numpy.unique(self.labels))
        if self.nobjects != 0:
            self.range = numpy.arange(1, numpy.max(self.labels) + 1)
            self.current_mean = scipy.ndimage.mean(self.image, self.labels)
            self.start_mean = numpy.maximum(self.current_mean, numpy.finfo(float).eps)


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
            image_set_loader.image_set_dict[channel],
            image_set_loader.image_set_dict[compartment],
            channel,
            compartment,
        )
        object_measurements = measure_3D_granularity_gpu(
            object_loader=object_loader,
            image_set_loader=image_set_loader,
            radius=20,
            granular_spectrum_length=16,
            subsample_size=0.25,
            image_name=object_loader.channel,
        )
        final_df = pd.DataFrame(object_measurements)
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
