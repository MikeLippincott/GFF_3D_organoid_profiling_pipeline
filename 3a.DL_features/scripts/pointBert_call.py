#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


import pathlib
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import tifffile
from mpl_toolkits.mplot3d import Axes3D
from pointbert_featurizer import MicroscopePointCloudFeaturePipeline

# ## Load a volume as a test

# In[ ]:


# save the point cloud
output_pcd_path = pathlib.Path(
    "../../point_clouds/C4-2_point_cloud.parquet"  # run from point-BERT directory
).resolve()
# read in the point cloud into a dictionary
point_clouds = {"coords": [], "features": [], "points": []}
df = pd.read_parquet(output_pcd_path)

point_clouds["coords"] = df[["x_points", "y_coords", "z_coords"]].to_numpy()
point_clouds["features"] = df.drop(columns=["feature_1", "feature_2"]).to_numpy()
point_clouds["points"] = df[["x_points", "y_coords", "z_coords"]].to_numpy()
point_clouds.keys()


# ## Point cloud conversion
#

# In[ ]:


checkpoint_url = "https://cloud.tsinghua.edu.cn/f/202b29805eea45d7be92/?dl=1"
checkpoint_path = pathlib.Path("../../models/point_bert.pth").resolve()
if not checkpoint_path.exists():
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(checkpoint_url, str(checkpoint_path))


# In[ ]:


# Initialize pipeline
pipeline = MicroscopePointCloudFeaturePipeline(pointbert_model_path=checkpoint_path)

# Extract features from single point cloud
features = pipeline.extract_features(point_clouds["points"])  # (384,)

print("Extracted features shape:", features.shape)
# Batch processing
# features = pipeline.extract_features_batch(point_clouds)  # (N, 384)


# In[ ]:


features_df = pd.DataFrame(features)
print(features_df.head())
