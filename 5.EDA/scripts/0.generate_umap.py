#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import numpy as np
import pandas as pd
import umap

# In[2]:


# paths to data
data_dict = {
    "NF0014": {
        "organoid_feature_selection": {
            "input_path": pathlib.Path(
                "../../data/NF0014/image_based_profiles/3.organoid_fs_profiles.parquet"
            ),
            "output_path": pathlib.Path(
                "../results/NF0014/3.organoid_fs_profiles_umap.parquet"
            ),
        },
        # "sc_feature_selection": {
        #     "input_path": pathlib.Path(
        #         "../../data/NF0014/image_based_profiles/3.sc_fs_profiles.parquet"
        #     ),
        #     "output_path": pathlib.Path(
        #         "../results/NF0014/3.sc_fs_profiles_umap.parquet"
        #     ),
        # },
        # "sc_consensus_profiles": {
        #     "input_path": pathlib.Path(
        #         "../../data/NF0014/image_based_profiles/4.sc_consensus_profiles.parquet"
        #     ),
        #     "output_path": pathlib.Path(
        #         "../results/NF0014/4.sc_consensus_profiles_umap.parquet"
        #     ),
        # },
        # "organoid_consensus_profiles": {
        #     "input_path": pathlib.Path(
        #         "../../data/NF0014/image_based_profiles/4.organoid_consensus_profiles.parquet"
        #     ),
        #     "output_path": pathlib.Path(
        #         "../results/NF0014/4.organoid_consensus_profiles_umap.parquet"
        #     ),
        # },
    }
}
pathlib.Path("../results/NF0014").mkdir(parents=True, exist_ok=True)


# In[3]:


metadata_columns = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "image_set",
    "Well",
    "single_cell_count",
    "parent_organoid",
    "MOA",
]


# In[4]:


umap_object = umap.UMAP(
    n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0
)

for dataset, paths in data_dict.items():
    for profile_type, paths in paths.items():
        print(f"Processing {dataset} - {profile_type}")

        # Load the data
        df = pd.read_parquet(paths["input_path"])

        metadata_df = df.copy()
        metadata_subset = []
        for col in metadata_columns:
            if col in df.columns:
                metadata_subset.append(col)

        metadata_df = df[metadata_subset]
        features_df = df.drop(columns=metadata_columns, errors="ignore")
        print(features_df.shape)
        # remove NaN values
        features_df = features_df.dropna(axis=1)
        print(f"Data shape after dropping NaN values: {features_df.shape}")
        # Extract features and apply UMAP

        umap_embedding = umap_object.fit_transform(features_df)

        # Create a DataFrame with UMAP results
        umap_df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2"])
        umap_df = pd.concat([metadata_df.reset_index(drop=True), umap_df], axis=1)
        # Save the UMAP results
        umap_df.to_parquet(paths["output_path"], index=False)
