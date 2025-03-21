#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys
import time

sys.path.append("../featurization")
from area_size_shape import measure_3D_area_size_shape
from colocalization import (
    measure_3D_colocalization,
    prepare_two_images_for_colocalization,
)
from granularity import measure_3D_granularity, measure_3D_granularity_gpu
from intensity import measure_3D_intensity
from loading_classes import ImageSetLoader, ObjectLoader, TwoObjectLoader
from neighbors import measure_3D_number_of_neighbors
from texture import measure_3D_texture

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
import gc
import itertools

import pandas as pd


# In[2]:


# begin profiling timer
start_whole_featurize = time.time()


# ### Set the path to the images 

# In[3]:


image_set_path = pathlib.Path("../../data/NF0014/cellprofiler/C4-2/")


# ### set the channel mapping dictionary

# In[4]:


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


# ### Initialize the image set loader

# In[5]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)
image_set_loader.image_set_dict.keys()


# ### Loop through the image set

# Run the rest in a script as it takes a long time to run

# In[6]:


texture = False


# In[7]:


gpu = False


# In[8]:


for compartment in tqdm(
    image_set_loader.compartments, desc="Processing compartments", position=0
):
    for channel in tqdm(
        image_set_loader.image_names,
        desc="Processing channels",
        leave=False,
        position=1,
    ):
        # compartment = "Organoid"
        # channel = "AGP"
        print("loading object loader")
        object_loader = ObjectLoader(
            image_set_loader.image_set_dict[channel],
            image_set_loader.image_set_dict[compartment],
            channel,
            compartment,
        )
        print(f"Compartment: {compartment}, Channel: {channel}")
        print("area size shape")
        start_time = time.time()
        # area, size, shape
        size_shape_dict = measure_3D_area_size_shape(
            image_set_loader=image_set_loader,
            object_loader=object_loader,
        )
        print(f"area size shape took {time.time() - start_time} seconds")
        start_time = time.time()
        # print("granularity")
        # # granularity
        # if gpu:
        #     object_measurements = measure_3D_granularity_gpu(
        #         object_loader,
        #         image_set_loader,
        #         radius=10,
        #         granular_spectrum_length=16,
        #         subsample_size=0.25,
        #         image_name=channel,
        #     )
        # else:
        #     object_measurements = measure_3D_granularity(
        #         object_loader,
        #         radius=10,
        #         granular_spectrum_length=16,
        #         subsample_size=0.25,
        #         image_name=channel,
        #     )
        # print(f"granularity took {time.time() - start_time} seconds")
        start_time = time.time()
        print("intensity")
        # intensity
        output_dict = measure_3D_intensity(object_loader)
        print(f"intensity took {time.time() - start_time} seconds")
        start_time = time.time()
        print("neighbors")
        # neighbors
        neighbors_out_dict = measure_3D_number_of_neighbors(
            object_loader=object_loader,
            distance_threshold=10,
            anisotropy_factor=image_set_loader.anisotropy_factor,
        )
        print(f"neighbors took {time.time() - start_time} seconds")
        if texture:
            start_time = time.time()

            output_texture_dict = measure_3D_texture(
                object_loader=object_loader,
                distance=1,
            )
            print(f"texture took {time.time() - start_time} seconds")
        print("merging")
        start_time = time.time()

        # merge the dataframes together
        size_shape_df = pd.DataFrame(size_shape_dict)
        # prepend the feature_type to the column names
        size_shape_df.columns = ["object_id"] + [
            "AreaSizeShape_" + col
            for col in size_shape_df.columns
            if col != "object_id"
        ]
        # granularity_df = pd.DataFrame(object_measurements)
        # # pivot wide
        # granularity_df = granularity_df.pivot(
        #     index="object_id", columns="feature", values="value"
        # )
        # granularity_df.reset_index(inplace=True)
        # # prepend the feature_type to the column names
        # granularity_df.columns = ["object_id"] + [
        #     "Granularity_" + col for col in granularity_df.columns if col != "object_id"
        # ]
        intensity_df = pd.DataFrame(output_dict)
        # pivot wide
        intensity_df = intensity_df.pivot(
            index="object_id", columns="feature_name", values="value"
        )
        intensity_df.reset_index(inplace=True)
        # prepend the feature_type to the column names
        intensity_df.columns = ["object_id"] + [
            "Intensity_" + col for col in intensity_df.columns if col != "object_id"
        ]
        neighbors_df = pd.DataFrame(neighbors_out_dict)
        # prepend the feature_type to the column names
        neighbors_df.columns = ["object_id"] + [
            "Neighbors_" + col for col in neighbors_df.columns if col != "object_id"
        ]
        # final_df = pd.merge(
        #     size_shape_df, granularity_df, left_on="object_id", right_on="object_id"
        # )
        final_df = pd.merge(
            size_shape_df, intensity_df, left_on="object_id", right_on="object_id"
        )
        # final_df = pd.merge(
        #     final_df, intensity_df, left_on="object_id", right_on="object_id"
        # )
        final_df = pd.merge(
            final_df, neighbors_df, left_on="object_id", right_on="object_id"
        )
        # prepend compartment and channel to column names
        final_df.columns = [
            f"{compartment}_{channel}_{col}" for col in final_df.columns
        ]
        final_df["image_set"] = image_set_loader.image_set_name
        print(final_df.shape)
        output_file = pathlib.Path(
            f"../results/{image_set_loader.image_set_name}_{compartment}_features.parquet"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_file)
        print("Merging took", time.time() - start_time)

        # remove the objects initialized in the beginning of the loop
        del object_loader
        del size_shape_dict
        del object_measurements
        del output_dict
        del neighbors_out_dict
        if texture:
            del output_texture_dict
        del final_df
        gc.collect()


# ## Run Colocalization

# In[ ]:


# get all channel combinations
channel_combinations = list(itertools.combinations(image_set_loader.image_names, 2))
output_list_of_dfs = []
for compartment in tqdm(
    image_set_loader.compartments, desc="Processing compartments", position=0
):
    for channel1, channel2 in tqdm(
        channel_combinations,
        desc="Processing channel combinations",
        leave=False,
        position=1,
    ):
        # compartment = "Organoid"
        # channel1 = "AGP"
        # channel2 = "ER"

        coloc_loader = TwoObjectLoader(
            image_set_loader=image_set_loader,
            compartment=compartment,
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
            # prepend compartment channel1 and channel2 to column names + colocalization
            coloc_df.columns = [
                f"{compartment}_{channel1}_{channel2}_Colocalization_" + col
                for col in coloc_df.columns
                if col != "object_id"
            ] + ["object_id"]
            output_list_of_dfs.append(coloc_df)
        coloc_df = pd.concat(output_list_of_dfs)
        coloc_df["image_set"] = image_set_loader.image_set_name
        print(coloc_df.shape)
output_file = pathlib.Path(
    f"../results/{image_set_loader.image_set_name}_{compartment}_coloc_features.parquet"
)
output_file.parent.mkdir(parents=True, exist_ok=True)
final_df = pd.concat(output_list_of_dfs)
final_df.to_parquet(output_file)


# In[ ]:


end = time.time()
print(
    f"Time taken for {image_set_loader.image_set_name} featurization:",
    end - start_whole_featurize,
)

