#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
import tqdm

# In[2]:


def calcualte_snr(image: np.ndarray) -> float:
    """
    Calculate the signal to noise ratio of an image
    Where SNR = mean(signal) / std(noise)
    Where the signal is the whole image pixel values

    Parameters
    ----------
    image_path : pathlib.Path
        Path to the image file

    Returns
    -------
    float
        Signal to noise ratio
    """
    p_sig = np.mean(image)
    p_noise = np.std(image)
    snr = p_sig / p_noise
    return snr


# In[3]:


images_path = pathlib.Path("../../data/NF0014/zstack_images").resolve(strict=True)
# get all the tif files recursively
tif_files = list(images_path.rglob("*.tif"))
test_file = tif_files[0]


# In[4]:


output_dict = {"well": [], "channel": [], "z-slice": [], "SNR": []}
for path_of_image in tqdm.tqdm(tif_files):
    well = path_of_image.stem.split("_")[0]
    channel = path_of_image.stem.split("_")[1]
    image = tifffile.imread(path_of_image)
    for z_slice in range(image.shape[0]):
        snr = calcualte_snr(image[z_slice])
        output_dict["well"].append(well)
        output_dict["channel"].append(channel)
        output_dict["z-slice"].append(z_slice)
        output_dict["SNR"].append(snr)


# In[5]:


df = pd.DataFrame(output_dict)
print(df.shape)
df.head()


# In[6]:


# save the SNR data to a csv file
pathlib.Path("../results").mkdir(exist_ok=True)
df.to_csv("../results/snr_data.csv", index=False)


# Typical SNR values
# * Low signal/quality confocal images (and STED): SNR = 5-10
# * Average confocal image: SNR = 15-20
# * High quality confocal image: SNR = > 30
# * Low quality widefield image (slidescanners): SNR 5-15
# * Good quality widefield image: SNR > 40
# * Microscope with cooled CCD camera: SNR = 50-100
