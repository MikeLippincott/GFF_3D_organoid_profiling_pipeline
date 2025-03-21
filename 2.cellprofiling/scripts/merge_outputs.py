#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import numpy as np
import pandas as pd

# In[13]:


path_to_output_files = pathlib.Path("../profiles_features").resolve()
# get all files in the directory
files = path_to_output_files.glob("*")
files = [x for x in files if x.is_file()]
# files = files[:10]


# In[27]:


df = pd.DataFrame(columns=["filename"], data=[x.stem for x in files])
df[["well_fov", "channel", "compartment", "mask", "object"]] = df["filename"].str.split(
    "_", expand=True
)
df.drop(columns=["mask"], inplace=True)
# group the data by well_fov, object
df.groupby(["well_fov", "object"]).agg(lambda x: x.tolist()).reset_index()


# In[ ]:
