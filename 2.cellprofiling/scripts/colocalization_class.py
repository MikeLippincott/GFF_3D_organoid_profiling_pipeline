#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
import skimage
import skimage.io as io
import skimage.morphology
import skimage.segmentation
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from scipy.linalg import lstsq

sys.path.append("../featurization")
import numpy
from area_size_shape import calulate_surface_area, measure_3D_area_size_shape
from colocalization import calculate_3D_colocalization
from data_writer import organize_featurization_data
from granularity import measure_3D_granularity
from intensity import measure_3D_intensity
from loading_classes import Featurization, ImageSetLoader, ObjectLoader
from neighbors import measure_3D_number_of_neighbors
from texture import measure_3D_texture
from two_object_loading_classes import (
    ColocalizationFeaturization,
    ColocalizationTwoObject_Loader,
)


# In[2]:


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


# In[3]:


image_set_path = pathlib.Path("../../data/NF0014/cellprofiler/C4-2/")


# In[4]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[5]:


# get all combinations of channels, compartments, and objects
unique_object = 4
colocalization_channel_pairs = []

for channel1 in image_set_loader.image_names:
    for channel2 in image_set_loader.image_names:
        if channel1 != channel2:
            colocalization_channel_pairs.append((channel1, channel2))

for compartment in image_set_loader.compartments:
    for unique_object in image_set_loader.unique_compartment_objects[compartment][1:2]:
        for channel1, channel2 in colocalization_channel_pairs:
            two_object_loader = ColocalizationTwoObject_Loader(
                image_set_loader=image_set_loader,
                image1=image_set_loader.image_set_dict[channel1],
                label_image1=image_set_loader.image_set_dict[compartment],
                object1=unique_object,
                image2=image_set_loader.image_set_dict[channel2],
                label_image2=image_set_loader.image_set_dict[compartment],
                object2=unique_object,
                compartment=compartment,
            )

            featurizer = ColocalizationFeaturization(
                image_set_loader=image_set_loader,
                two_object_loader=two_object_loader,
            )
            features = featurizer.process_features_for_output()
            organize_featurization_data(
                features=features,
                compartment=compartment,
                channel=f"{channel1}-{channel2}",
                label_index=unique_object,
                output=True,
                output_path=pathlib.Path(
                    f"../profiles_features/{image_set_loader.image_set_name}_{compartment}_{channel1}_{channel2}_{unique_object}.parquet"
                ),
                image_set_name=image_set_loader.image_set_name,
                return_df=False,
            )


# In[ ]:


pd.read_parquet("../profiles_features/C4-2_cell_mask_AGP_ER_4.parquet")

