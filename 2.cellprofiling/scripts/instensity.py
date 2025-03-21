#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

sys.path.append("../featurization")
import numpy as np
import pandas as pd
import scipy
import skimage
from data_writer import organize_featurization_data
from intensity import measure_3D_intensity
from loading_classes import ImageSetLoader, ObjectLoader

# In[2]:


image_set_path = pathlib.Path("../../data/NF0014/cellprofiler/C4-2/")


# In[3]:


channel_mapping = {
    "nuclei": "405",
    "AGP": "488",
    "ER": "555",
    "Mito": "640",
    "BF": "TRANS",
    "nuclei_mask": "nuclei_",
    "cell_mask": "cell_",
    "cytoplasm_mask": "cytoplasm_",
    "organoid_mask": "organoid_",
}


# In[4]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[5]:


object_loader = ObjectLoader(
    image_set_loader.image_set_dict["nuclei"],
    image_set_loader.image_set_dict["nuclei_mask"],
    "nuclei",
    "nuclei",
)


# In[6]:


output_dict = measure_3D_intensity(object_loader)


# In[7]:


df = pd.DataFrame(output_dict)
df.head()
