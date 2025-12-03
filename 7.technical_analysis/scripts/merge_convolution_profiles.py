#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from notebook_init_utils import bandicoot_check, init_notebook
from pycytominer import feature_select, normalize

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


convolutions = list(range(1, 26))
convolutions += [50, 75, 100]
subdir_names = [f"convolution_{i}_image_based_profiles" for i in convolutions]
subdir_names += ["image_based_profiles"]
subdir_names += ["deconvolved_images_image_based_profiles"]


# In[3]:


organoid_convolution_paths = [
    pathlib.Path(
        f"{image_base_dir}/data/NF0014_T1/{subdir_name}/2.annotated_profiles/organoid_anno.parquet"
    )
    for subdir_name in subdir_names
]

sc_convolution_paths = [
    pathlib.Path(
        f"{image_base_dir}/data/NF0014_T1/{subdir_name}/2.annotated_profiles/sc_anno.parquet"
    )
    for subdir_name in subdir_names
]


# In[4]:


# Merge organoid profiles
organoid_combined_dfs = pd.concat(
    [
        pd.read_parquet(path).assign(
            Metadata_convolution=path.parent.parent.name.replace(
                "_image_based_profiles", ""
            )
        )
        for path in organoid_convolution_paths
    ],
    ignore_index=True,
)
organoid_combined_dfs["Metadata_convolution"] = organoid_combined_dfs[
    "Metadata_convolution"
].replace(
    {
        "image_based_profiles": "0",
        "deconvolved_images": "-1",
        "convolution_": "",
    },
    regex=True,
)
# drop all Metadata_image_sets that are not C4-2
organoid_combined_dfs = organoid_combined_dfs[
    organoid_combined_dfs["Metadata_image_set"] == "C4-2"
]
organoid_combined_dfs.reset_index(drop=True, inplace=True)
organoid_combined_dfs["Metadata_convolution"] = organoid_combined_dfs[
    "Metadata_convolution"
].astype(str)

# merge sc profiles
sc_combined_dfs = pd.concat(
    [
        pd.read_parquet(path).assign(
            Metadata_convolution=path.parent.parent.name.replace(
                "_image_based_profiles", ""
            )
        )
        for path in sc_convolution_paths
    ],
    ignore_index=True,
)
sc_combined_dfs["Metadata_convolution"] = sc_combined_dfs[
    "Metadata_convolution"
].replace(
    {
        "image_based_profiles": "0",
        "deconvolved_images": "-1",
        "convolution_": "",
    },
    regex=True,
)
# drop all Metadata_image_sets that are not C4-2
sc_combined_dfs = sc_combined_dfs[sc_combined_dfs["Metadata_image_set"] == "C4-2"]
sc_combined_dfs.reset_index(drop=True, inplace=True)
sc_combined_dfs["Metadata_convolution"] = sc_combined_dfs[
    "Metadata_convolution"
].astype(str)


# In[5]:


organoid_metadata_columns = [
    x for x in organoid_combined_dfs.columns if "Metadata" in x
]

organoid_metadata_columns += [
    "Area.Size.Shape_Organoid_CENTER.X",
    "Area.Size.Shape_Organoid_CENTER.Y",
    "Area.Size.Shape_Organoid_CENTER.Z",
]
organoid_features_columns = [
    col for col in organoid_combined_dfs.columns if col not in organoid_metadata_columns
]

# sc metadata and features
sc_metadata_columns = [x for x in sc_combined_dfs.columns if "Metadata" in x]

sc_metadata_columns += [
    "Area.Size.Shape_Cell_CENTER.X",
    "Area.Size.Shape_Cell_CENTER.Y",
    "Area.Size.Shape_Cell_CENTER.Z",
    "Area.Size.Shape_Nuclei_CENTER.X",
    "Area.Size.Shape_Nuclei_CENTER.Y",
    "Area.Size.Shape_Nuclei_CENTER.Z",
    "Area.Size.Shape_Cytoplasm_CENTER.X",
    "Area.Size.Shape_Cytoplasm_CENTER.Y",
    "Area.Size.Shape_Cytoplasm_CENTER.Z",
]
sc_features_columns = [
    col for col in sc_combined_dfs.columns if col not in sc_metadata_columns
]


# In[7]:


# normalize the data
organoid_normalized_profiles = normalize(
    organoid_combined_dfs,
    features=organoid_features_columns,
    meta_features=organoid_metadata_columns,
    method="standardize",
    samples="Metadata_convolution == '0'",
)
organoid_fs_df = feature_select(
    organoid_normalized_profiles,
    features=organoid_features_columns,
)

# normalize the data sc
sc_normalized_profiles = normalize(
    sc_combined_dfs,
    features=sc_features_columns,
    meta_features=sc_metadata_columns,
    method="standardize",
    samples="Metadata_convolution == '0'",
)
sc_fs_df = feature_select(
    sc_normalized_profiles,
    features=sc_features_columns,
)


# In[8]:


organoid_metadata_columns = [x for x in organoid_fs_df.columns if "Metadata" in x]

organoid_metadata_columns += [
    "Area.Size.Shape_Organoid_CENTER.X",
    "Area.Size.Shape_Organoid_CENTER.Y",
    "Area.Size.Shape_Organoid_CENTER.Z",
]
organoid_features_columns = [
    col for col in organoid_fs_df.columns if col not in organoid_metadata_columns
]

# sc metadata and features
sc_metadata_columns = [x for x in sc_fs_df.columns if "Metadata" in x]
sc_metadata_columns += [
    "Area.Size.Shape_Cell_CENTER.X",
    "Area.Size.Shape_Cell_CENTER.Y",
    "Area.Size.Shape_Cell_CENTER.Z",
    "Area.Size.Shape_Nuclei_CENTER.X",
    "Area.Size.Shape_Nuclei_CENTER.Y",
    "Area.Size.Shape_Nuclei_CENTER.Z",
    "Area.Size.Shape_Cytoplasm_CENTER.X",
    "Area.Size.Shape_Cytoplasm_CENTER.Y",
    "Area.Size.Shape_Cytoplasm_CENTER.Z",
]
sc_features_columns = [
    col for col in sc_fs_df.columns if col not in sc_metadata_columns
]


# In[9]:


# save the normalized profiles
organoid_normalized_output_path = pathlib.Path(
    f"{root_dir}/7.technical_analysis/processed_data/organoid_fs_convolution_profiles.parquet"
).resolve()
sc_normalized_output_path = pathlib.Path(
    f"{root_dir}/7.technical_analysis/processed_data/sc_fs_convolution_profiles.parquet"
).resolve()
organoid_normalized_output_path.parent.mkdir(parents=True, exist_ok=True)
organoid_fs_df = organoid_fs_df.dropna(subset=organoid_features_columns)
organoid_fs_df.to_parquet(organoid_normalized_output_path, index=False)
sc_normalized_output_path.parent.mkdir(parents=True, exist_ok=True)
sc_fs_df = sc_fs_df.dropna(subset=sc_features_columns)
sc_fs_df.to_parquet(sc_normalized_output_path, index=False)


# In[ ]:


# plot on UMAP (use continuous colorbar from -1 to 25)
organoid_fs_df["Metadata_convolution"] = organoid_fs_df["Metadata_convolution"].astype(
    int
)
# drop NaN values
reducer = umap.UMAP(random_state=0)
embedding = reducer.fit_transform(organoid_fs_df[organoid_features_columns].values)

cvals = organoid_fs_df["Metadata_convolution"].astype(float).values
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=cvals,
    cmap="viridis",
    vmin=-1,
    s=10,
    alpha=0.7,
)
cbar = plt.colorbar(sc)
cbar.set_label("Convolution")
cbar.set_ticks([-1, 0, 5, 10, 15, 20, 25, 50, 75, 100])

plt.title("UMAP of Normalized Organoid Profiles")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()


# In[11]:


# plot on UMAP (use continuous colorbar from -1 to 25)
sc_fs_df["Metadata_convolution"] = sc_fs_df["Metadata_convolution"].astype(int)
# drop NaN values
sc_fs_df = sc_fs_df.dropna(subset=sc_features_columns)
reducer = umap.UMAP(random_state=0)
embedding = reducer.fit_transform(sc_fs_df[sc_features_columns].values)

cvals = sc_fs_df["Metadata_convolution"].astype(float).values
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=cvals,
    cmap="viridis",
    vmin=-1,
    s=10,
    alpha=0.7,
)
cbar = plt.colorbar(sc)
cbar.set_label("Convolution")
cbar.set_ticks([-1, 0, 5, 10, 15, 20, 25, 50, 75, 100])

plt.title("UMAP of Normalized Organoid Profiles")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()
