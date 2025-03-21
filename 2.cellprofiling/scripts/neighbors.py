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
from loading_classes import ImageSetLoader, ObjectLoader
from neighbors import measure_3D_number_of_neighbors

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


label_object = object_loader.objects
label_image = object_loader.label_image
distance_threshold = 10
anisotropy_factor = 10
labels = object_loader.object_ids


# In[7]:


neighbors_out_dict = measure_3D_number_of_neighbors(
    object_loader=object_loader,
    distance_threshold=distance_threshold,
    anisotropy_factor=anisotropy_factor,
)


# In[8]:


df = pd.DataFrame(neighbors_out_dict)
df.head()
