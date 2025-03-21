#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

sys.path.append("../featurization")
import itertools

import numpy as np
import pandas as pd
import scipy
import skimage
from colocalization import (
    measure_3D_colocalization,
    prepare_two_images_for_colocalization,
)
from data_writer import organize_featurization_data
from loading_classes import ImageSetLoader, ObjectLoader
from two_object_loading_classes import TwoObjectLoader

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


# get all channel combinations
channel_combinations = list(itertools.combinations(image_set_loader.image_names, 2))


# In[6]:


output_list_of_dfs = []
for compartments in tqdm(
    image_set_loader.compartments, desc="Processing compartments", position=0
):
    for channel1, channel2 in tqdm(
        channel_combinations,
        desc="Processing channel combinations",
        leave=False,
        position=1,
    ):
        coloc_loader = TwoObjectLoader(
            image_set_loader=image_set_loader,
            compartment=compartments,
            channel1=channel1,
            channel2=channel2,
        )
        for object_id in tqdm(
            coloc_loader.object_ids,
            desc="Processing object IDs",
            leave=False,
            position=2,
        ):
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
            coloc_df = pd.DataFrame(colocalization_features, index=[0])
            coloc_df["object_id"] = object_id
            coloc_df["channel1"] = channel1
            coloc_df["channel2"] = channel2
            coloc_df["compartment"] = compartments
            output_list_of_dfs.append(coloc_df)


# In[7]:


final_df = pd.concat(output_list_of_dfs, ignore_index=True)
final_df.head()
