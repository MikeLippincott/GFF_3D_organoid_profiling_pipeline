#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import pathlib
import sys
import urllib.request

import tifffile

sys.path.append(str("../utils/"))
from sammed3d_featurizer import MicroscopySAMMed3DPipeline

# ## Load a volume as a test

# In[2]:


# load a volume as a test
nuclei_volume_path = pathlib.Path(
    "/home/lippincm/mnt/bandicoot/NF1_organoid_data/data/NF0014_T1/zstack_images/C4-2/C4-2_405.tif"
).resolve(strict=True)
volume = tifffile.imread(nuclei_volume_path)


# In[3]:


sam3dmed_checkpoint_url = (
    "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
)
sam3dmed_checkpoint_path = pathlib.Path("../models/sam-med3d-turbo.pth").resolve()
if not sam3dmed_checkpoint_path.exists():
    sam3dmed_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(sam3dmed_checkpoint_url, str(sam3dmed_checkpoint_path))


# In[4]:


# Initialize with pretrained weights
pipeline = MicroscopySAMMed3DPipeline(
    sammed3d_path=str(sam3dmed_checkpoint_path),  # Auto-downloads SAM-Med3D-turbo
    feature_type="global",
)

# Extract features from single volume
features = pipeline.extract_features(volume)  # Shape: (768,)

# # Batch processing
# features = pipeline.extract_features_batch(volumes)  # Shape: (N, 768)
print(features.shape)
