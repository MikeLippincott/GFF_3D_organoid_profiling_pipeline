#!/usr/bin/env python
# coding: utf-8

# The goal of this notebook is to reassign segmentation labels based on the objects that they are contained in.
# To do so, a hierarchy of objects must first be defined.
# # The hierarchy of objects is defined as follows:
# - **Cell**
#     - **Nucleus**
#     - **Cytoplasm**
#
# The index of a given cytoplasm should be the same as that of cell it came from.
# The nucleus index should be the same as that of the cell it came from.
#
# There will also be rules implemented for sandwiched indexes.
# This is when an object was not related properly and was assigned a different index while being surrounded (above and below in the z dimension) by the same object.
# Such cases will be assigned the same index as the object that is above and below it.
#

# In[1]:


import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import tifffile
from cellpose import core, models, utils
from rich.pretty import pprint

sys.path.append("../../utils")
import nviz
from nviz.image_meta import extract_z_slice_number_from_filename, generate_ome_xml
from segmentation_decoupling import euclidian_2D_distance

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


def centroid_within_bbox_detection(
    centroid: tuple,
    bbox: tuple,
) -> bool:
    """
    Check if the centroid is within the bbox

    Parameters
    ----------
    centroid : tuple
        Centroid of the object in the order of (z, y, x)
        Order of the centroid is important
    bbox : tuple
        Where the bbox is in the order of (z_min, y_min, x_min, z_max, y_max, x_max)
        Order of the bbox is important

    Returns
    -------
    bool
        True if the centroid is within the bbox, False otherwise
    """
    z_min, y_min, x_min, z_max, y_max, x_max = bbox
    z, y, x = centroid
    # check if the centroid is within the bbox
    if (
        z >= z_min
        and z <= z_max
        and y >= y_min
        and y <= y_max
        and x >= x_min
        and x <= x_max
    ):
        return True
    else:
        return False


# In[3]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--patient",
        type=str,
        help="The patient ID",
    )

    parser.add_argument(
        "--well_fov",
        type=str,
        help="Path to the input directory containing the tiff images",
    )
    parser.add_argument(
        "--radius_constraint",
        type=int,
        default=10,
        help="The maximum radius of the x-y vector",
    )
    parser.add_argument(
        "--compartment",
        type=str,
        default="none",
        help="The compartment to segment",
    )

    args = parser.parse_args()
    well_fov = args.well_fov
    x_y_vector_radius_max_constaint = args.radius_constraint
    compartment = args.compartment
    patient = args.patient
else:
    print("Running in a notebook")
    well_fov = "C2-2"
    compartment = "nuclei"
    patient = "NF0014"

mask_dir = pathlib.Path(f"../../data/{patient}/processed_data/{well_fov}").resolve()


# In[ ]:


# get the organoid masks
cell_mask_path = mask_dir / "cell_masks_reconstructed.tiff"
nuclei_mask_path = mask_dir / "nuclei_masks_reconstructed.tiff"
cell_mask = io.imread(cell_mask_path)
nuclei_mask = io.imread(nuclei_mask_path)


# In[5]:


# get the first z slice of the image
for z in range(cell_mask.shape[0]):
    plt.subplot(1, 2, 1)
    plt.imshow(cell_mask[z, :, :], cmap="magma")
    plt.title("Cell mask")
    plt.subplot(1, 2, 2)
    plt.imshow(nuclei_mask[z, :, :], cmap="magma")
    plt.title("Nuclei mask")
    plt.show()


# In[14]:


# get the centroid and bbox of the cell mask
cell_df = pd.DataFrame.from_dict(
    skimage.measure.regionprops_table(
        cell_mask,
        properties=["centroid", "bbox"],
    )
)
cell_df["compartment"] = "cell"
cell_df["label"] = cell_mask[
    cell_df["centroid-0"].astype(int),
    cell_df["centroid-1"].astype(int),
    cell_df["centroid-2"].astype(int),
]
# remove all 0 labels
cell_df = cell_df[cell_df["label"] > 0].reset_index(drop=True)
cell_df["new_label"] = cell_df["label"]


# In[15]:


nuclei_df = pd.DataFrame.from_dict(
    skimage.measure.regionprops_table(
        nuclei_mask,
        properties=["centroid", "bbox"],
    )
)
nuclei_df["compartment"] = "nuclei"
nuclei_df["label"] = nuclei_mask[
    nuclei_df["centroid-0"].astype(int),
    nuclei_df["centroid-1"].astype(int),
    nuclei_df["centroid-2"].astype(int),
]
nuclei_df = nuclei_df[nuclei_df["label"] > 0].reset_index(drop=True)
nuclei_df["new_label"] = nuclei_df["label"]
nuclei_df


# In[16]:


# if a centroid of the nuclei is inside the cell mask,
# then make the cell retain the label of the nuclei
for i, row in nuclei_df.iterrows():
    for j, row2 in cell_df.iterrows():
        nuc_contained_in_cell_bool = centroid_within_bbox_detection(
            centroid=(
                row["centroid-0"],
                row["centroid-1"],
                row["centroid-2"],
            ),
            bbox=(
                row2["bbox-0"],
                row2["bbox-1"],
                row2["bbox-2"],
                row2["bbox-3"],
                row2["bbox-4"],
                row2["bbox-5"],
            ),
        )
        if nuc_contained_in_cell_bool:
            # if the centroid of the nuclei is within the cell mask,
            # then make the cell retain the label of the nuclei
            cell_df.at[j, "new_label"] = row["label"]
            # print(f"Cell {row2['label']} contains nuclei {row['label']}")
            break
        else:
            # print(f"Cell {row2['label']} does not contain nuclei {row['label']}")
            pass


# In[17]:


nuclei_df


# In[18]:


cell_df


# In[ ]:


def mask_label_reassignment(
    mask_df: pd.DataFrame,
    mask_input: np.ndarray,
    compartment: str = "none",
) -> np.ndarray:
    """
    Reassign the labels of the mask based on the mask_df

    Parameters
    ----------
    mask_df : pd.DataFrame
        DataFrame containing the labels and centroids of the mask
    mask_input : np.ndarray
        The input mask to reassign the labels to
    compartment : str, optional
        The compartment to segment, by default "none"

    Returns
    -------
    np.ndarray
        The mask with reassigned labels
    """
    for i, row in mask_df.iterrows():
        if row["label"] == row["new_label"]:
            # if the label is already the new label, skip
            continue
        mask_input[mask_input == row["label"]] = row["new_label"]


# In[19]:


mask_example = cell_mask.copy()

mask_example[mask_example == 12] = 24


# In[21]:


for z in range(mask_example.shape[0]):
    plt.subplot(1, 2, 1)
    plt.imshow(cell_mask[z, :, :], cmap="magma")
    plt.title("Original cell mask")
    plt.subplot(1, 2, 2)
    plt.imshow(mask_example[z, :, :], cmap="magma")
    plt.title("Modified cell mask")
    plt.show()


# In[ ]:
