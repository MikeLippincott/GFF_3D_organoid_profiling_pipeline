#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pathlib
import time

import numpy as np
import pandas as pd
import psutil
import tifffile
import torch
import tqdm
from arg_parsing_utils import parse_args
from notebook_init_utils import (
    avoid_path_crash_bandicoot,
    bandicoot_check,
    init_notebook,
)
from torch import rand
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)
print(image_base_dir)
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


# begin tiem and memory profiling
stat_time = time.time()
start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # in MB


# In[3]:


def calculate_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculate the Mean Squared Error (MSE) between two images."""
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    mse = np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
    return float(mse)


def calculate_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float("inf")  # No difference between images

    max_pixel = 255.0 if image1.dtype == np.uint8 else 65535.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_dists(image1: np.ndarray, image2: np.ndarray) -> float:
    """Calculate the DISTS metric between two images."""
    # Convert to float tensor and add batch & channel dims: [1, 1, X, Y]
    t1 = torch.from_numpy(image1).unsqueeze(0).unsqueeze(0).float()
    t2 = torch.from_numpy(image2).unsqueeze(0).unsqueeze(0).float()

    metric = DeepImageStructureAndTextureSimilarity()
    dists_metric = metric(t1, t2).item()
    return float(dists_metric)


# In[4]:


if not in_notebook:
    arguments_dict = parse_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    channel = arguments_dict["channel"]

else:
    well_fov = "D8-2"
    patient = "NF0016_T1"

channels = ["405", "488", "555", "640"]


# In[5]:


raw_file_file_paths = []
for channel in channels:
    raw_file_file_paths.append(
        pathlib.Path(
            f"{image_base_dir}/data/{patient}/zstack_images/{well_fov}/{well_fov}_{channel}.tif"
        ).resolve()
    )
output_dir = pathlib.Path("../results/decon_image_metrics/individual_files").resolve()
output_dir.mkdir(parents=True, exist_ok=True)


# In[6]:


df = pd.DataFrame({"image_path": raw_file_file_paths})
df["patient"] = df["image_path"].apply(lambda x: x.parent.parent.parent.name)
df["well_fov"] = df["image_path"].apply(lambda x: x.parent.name)
df["channel"] = df["image_path"].apply(lambda x: x.stem.split("_")[-1])

image_path = df.pop("image_path")
df.insert(3, "image_path", image_path)

# filter out rows that contain channel = TRANS
df = df[df["channel"] != "TRANS"].reset_index(drop=True)

# Ensure we pivot patient x well_fov -> one column per channel (values are the image_path)
df = df[["patient", "well_fov", "channel", "image_path"]].copy()
# convert paths to strings (optional)
df["image_path"] = df["image_path"].astype(str)

df["decon_image_path"] = df.apply(
    lambda row: pathlib.Path(
        f"{image_base_dir}/data/{row['patient']}/deconvolved_images/{row['well_fov']}/{row['well_fov']}_{row['channel']}.tif"
    ),
    axis=1,
)
df.head()


# In[7]:


for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    results_dict = {
        "patient": [],
        "well_fov": [],
        "channel": [],
        "zslice": [],
        "psnr": [],
        "dists": [],
    }

    output_file_dir = pathlib.Path(
        f"{output_dir}/{row['patient']}_{row['well_fov']}_{row['channel']}_decon_image_metrics.parquet"
    ).resolve()
    output_file_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_file_dir.exists():
        print(f"Output file {output_file_dir} already exists, skipping...")
        continue
    raw_image = tifffile.imread(row["image_path"])
    decon_image = tifffile.imread(row["decon_image_path"])
    if raw_image.shape != decon_image.shape:
        print(raw_image.shape, decon_image.shape)
        print(f"Image shape mismatch for index {idx}, skipping...")
        continue
    for zslice in range(raw_image.shape[0]):
        raw_z_slice = raw_image[zslice]
        decon_z_slice = decon_image[zslice]

        psnr = calculate_psnr(raw_z_slice, decon_z_slice)
        dists = calculate_dists(raw_z_slice, decon_z_slice)
        results_dict["patient"].append(row["patient"])
        results_dict["well_fov"].append(row["well_fov"])
        results_dict["channel"].append(row["channel"])
        results_dict["zslice"].append(zslice)
        results_dict["psnr"].append(psnr)
        results_dict["dists"].append(dists)
    pd.DataFrame(results_dict).to_parquet(output_file_dir)


# In[ ]:


end_time = time.time()
end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # in MB
print(f"Time elapsed: {end_time - stat_time} seconds")
print(f"Memory used: {end_mem - start_mem} MB")
