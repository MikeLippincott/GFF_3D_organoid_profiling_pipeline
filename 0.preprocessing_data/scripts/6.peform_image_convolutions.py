#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import sys

import matplotlib.pyplot as plt

# Import dependencies
import numpy as np
import skimage
import tifffile
from skimage import io

cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd
else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break
sys.path.append(str(root_dir / "utils"))
from arg_parsing_utils import check_for_missing_args, parse_args
from file_reading import read_zstack_image
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()
if in_notebook:
    from tqdm.notebook import tqdm
else:
    import tqdm

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


if not in_notebook:
    args = parse_args()
    window_size = args["window_size"]
    clip_limit = args["clip_limit"]
    well_fov = args["well_fov"]
    patient = args["patient"]
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
        window_size=window_size,
        clip_limit=clip_limit,
    )
else:
    print("Running in a notebook")
    patient = "NF0014_T1"
    well_fov = "C4-2"
    window_size = 3
    clip_limit = 0.05


input_dir_raw = pathlib.Path(
    f"{image_base_dir}/data/{patient}/zstack_images/{well_fov}"
).resolve(strict=True)
input_dir_decon = pathlib.Path(
    f"{image_base_dir}/data/{patient}/deconvolved_images/{well_fov}"
).resolve(strict=True)
mask_path_raw = pathlib.Path(
    f"{image_base_dir}/data/{patient}/segmentation_masks/{well_fov}"
).resolve(strict=True)
mask_path_decon = pathlib.Path(
    f"{image_base_dir}/data/{patient}/deconvolved_segmentation_masks/{well_fov}"
).resolve(strict=True)


# In[3]:


# read in the image
list_of_decon_images = list(input_dir_decon.glob("*.tif"))
list_of_raw_images = list(input_dir_raw.glob("*.tif"))
list_of_decon_images.sort()
list_of_raw_images.sort()


# In[4]:


image_dict = {
    "image": [],
    "labels": [],
}
convolutions = 100
convolution_step = 25


# In[5]:


for img_path in list_of_raw_images:
    if "trans" in img_path.name.lower():
        continue
    img = read_zstack_image(img_path)
    # convolve the image with a gaussian filter
    image_dict["image"].append(img)
    image_dict["labels"].append(f"raw_{img_path.stem.split('_')[1]}")

for img_path in list_of_decon_images:
    img = read_zstack_image(img_path)
    # convolve the image with a gaussian filter

    image_dict["image"].append(img)
    image_dict["labels"].append(f"deconvolved_raw_{img_path.stem.split('_')[1]}")
    for convolution in tqdm(range(1, convolutions + 1), desc="Convolving image"):
        img = skimage.filters.gaussian(img, sigma=3)

        if (convolution) % convolution_step == 0:
            convolution_output_path = pathlib.Path(
                f"{image_base_dir}/data/{patient}/convolution_{convolution}"
            ).resolve()
            convolution_output_path.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(convolution_output_path / f"{img_path.name}")
            image_dict["image"].append(img)
            image_dict["labels"].append(
                f"convolution_{convolution}_{img_path.stem.split('_')[1]}"
            )


# In[ ]:


plt.figure(figsize=(10, 10))
for i in range(image_dict["image"].__len__()):
    plt.subplot(6, image_dict["image"].__len__() // 6 + 1, i + 1)
    plt.title(f"{image_dict['labels'][i]}")
    plt.imshow(image_dict["image"][i][9], cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.show()
