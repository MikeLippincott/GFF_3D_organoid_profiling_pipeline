#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import pathlib
import sys
from itertools import product

import pandas as pd

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")

sys.path.append(f"{root_dir}/3.cellprofiling/featurization_utils/")
from loading_classes import ImageSetLoader

# In[ ]:


well_fov = "C4-2"
patient = "NF0014"
channel = "DNA"
compartment = "Nuclei"
processor_type = "CPU"

image_set_path = pathlib.Path(
    f"{root_dir}/data/{patient}/profiling_input_images/{well_fov}/"
)
input_combinations_path = pathlib.Path(
    f"{root_dir}/3.cellprofiling/load_data/input_combinations.json"
)
input_combinations_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


features = [
    "AreaSizeShape",
    "Colocalization",
    "Granularity",
    "Intensity",
    "Neighbors",
    "Texture",
]


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


# In[5]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[6]:


output_dict = {
    "feature": [],
    "compartment": [],
    "channel": [],
}


# In[7]:


for feature in features:
    if feature == "Neighbors":
        output_dict["feature"].append("Neighbors")
        output_dict["compartment"].append("Nuclei")
        output_dict["channel"].append("DNA")
    for compartment in image_set_loader.compartments:
        if feature == "AreaSizeShape":
            output_dict["feature"].append("AreaSizeShape")
            output_dict["compartment"].append(compartment)
            output_dict["channel"].append("DNA")
        for channel in image_set_loader.image_names:
            if (
                feature != "Neighbors"
                and feature != "AreaSizeShape"
                and feature != "Colocalization"
            ):
                output_dict["feature"].append(feature)
                output_dict["compartment"].append(compartment)
                output_dict["channel"].append(channel)


# In[8]:


# get all channel combinations
channel_combinations = list(itertools.combinations(image_set_loader.image_names, 2))


# In[9]:


combinations = [
    (compartment, channel1, channel2)
    for compartment, (channel1, channel2) in product(
        image_set_loader.compartments, channel_combinations
    )
]


# In[10]:


for combination in combinations:
    output_dict["feature"].append("Colocalization")
    output_dict["compartment"].append(combination[0])
    output_dict["channel"].append(f"{combination[1]}.{combination[2]}")
df = pd.DataFrame(output_dict)

# write to json
df.to_json(input_combinations_path, orient="records", indent=4)


# In[11]:


df.head()


# In[12]:


df


# In[ ]:
