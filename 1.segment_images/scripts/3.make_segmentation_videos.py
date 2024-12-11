#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import imageio
import skimage.io as io

# In[2]:


# set paths
nuclei_img_path = pathlib.Path("../examples/raw_z_input/C6-1_405.tif").resolve(
    strict=True
)
cell_img_path = pathlib.Path("../examples/raw_z_input/C6-1_555.tif").resolve(
    strict=True
)
nuclei_mask_path = pathlib.Path(
    "../../data/nuclei_masks/raw_z_input/nuclei_masks.tiff"
).resolve(strict=True)
cell_mask_path = pathlib.Path(
    "../../data/cell_masks/raw_z_input/cell_masks.tiff"
).resolve(strict=True)

output_path = pathlib.Path("../examples/segmentation_output").resolve()
output_path.mkdir(exist_ok=True, parents=True)


# In[3]:


# read in the nucei and cell masks
nuclei_img = io.imread(nuclei_img_path)
cell_img = io.imread(cell_img_path)
nuclei_mask = io.imread(nuclei_mask_path)
cell_mask = io.imread(cell_mask_path)

# scale the images to unit8
nuclei_img = (nuclei_img / 255).astype("uint8") * 4
cell_img = (cell_img / 255).astype("uint8") * 8
# make the images brighter
nuclei_mask = (nuclei_mask * 255).astype("uint8")
cell_mask = (cell_mask * 255).astype("uint8")


# ## Nuclei image visualization

# In[4]:


frames = [nuclei_img[i] for i in range(nuclei_img.shape[0])]

# Write the frames to a GIF
output_file_path = pathlib.Path(output_path / "nuclei_img_output.gif")
imageio.mimsave(output_file_path, frames, duration=0.1, loop=10)


# ## Cell image visualization

# In[5]:


frames = [cell_img[i] for i in range(cell_img.shape[0])]

# Write the frames to a GIF
output_file_path = pathlib.Path(output_path / "cell_img_output.gif")
imageio.mimsave(output_file_path, frames, duration=0.1, loop=10)


# ## Nuclei segmentation visualization

# In[6]:


frames = [nuclei_mask[i] for i in range(nuclei_mask.shape[0])]

# Write the frames to a GIF
output_file_path = pathlib.Path(output_path / "nuclei_mask_output.gif")
imageio.mimsave(output_file_path, frames, duration=0.1, loop=10)


# ## Cell segmentation visualization

# In[7]:


frames = [cell_mask[i] for i in range(cell_mask.shape[0])]

# Write the frames to a GIF
output_file_path = pathlib.Path(output_path / "cell_mask_output.gif")
imageio.mimsave(
    output_file_path, frames, duration=0.1, loop=10
)  # duration is the time between frames in seconds
