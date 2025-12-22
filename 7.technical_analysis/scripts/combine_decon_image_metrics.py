#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pathlib

import numpy as np
import pandas as pd
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


results_dir = pathlib.Path("../results/decon_image_metrics/individual_files").resolve(
    strict=True
)
combined_df_dir = pathlib.Path(
    "../results/decon_image_metrics/combined_decon_image_metrics.parquet"
).resolve()
list_of_files = list(results_dir.glob("*"))
df = pd.concat([pd.read_parquet(f) for f in list_of_files], ignore_index=True)
print(df.shape)
df.sort_values(by=["patient", "well_fov", "channel", "zslice"], inplace=True)
df.reset_index(inplace=True, drop=True)
df.head()


# In[3]:


# clean the data
# find inf values and make 0


def clean_data(df):
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df


df = clean_data(df)


# In[4]:


df.to_parquet(combined_df_dir, index=False)
