#!/usr/bin/env python
# coding: utf-8

# # Segmentation corrections

# The goal of this notebook is to correct potential errors in the segmentation of the 3D image data.
# Potential errors can be observed in the figure below where each row is a different slice of the 3D image data and each column is a different outcome of the segmentation.
# Each segmentation of cell and nucleus is shown in a different color.
# Where cells or nuclei that are the same object id are shown in the same color.
# While cells and nuclei that are different object ids are shown in different colors.
# Some of the outcomes are not correct and need to be corrected.
# While others might be correct or incorrect but there is not logical way to determine if they are correct or not.
# These cases are not corrected.
# ![Segmentation errors](../media/3D_segmentations_correction_events.png)
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
    well_fov = "C4-2"
    compartment = "cell"
    patient = "NF0014"

mask_dir = pathlib.Path(f"../../data/{patient}/processed_data/{well_fov}").resolve()


# In[3]:


if compartment == "nuclei":
    mask_path = pathlib.Path(mask_dir / "nuclei_masks_reconstructed.tiff").resolve(
        strict=True
    )
    mask_output_path = pathlib.Path(
        mask_dir / "masks_reconstructed_corrected.tiff"
    ).resolve()
elif compartment == "cell":
    mask_path = pathlib.Path(mask_dir / "cell_masks_reconstructed.tiff").resolve(
        strict=True
    )
    mask_output_path = pathlib.Path(
        mask_dir / "cell_masks_reconstructed_corrected.tiff"
    ).resolve()
elif compartment == "organoid":
    mask_path = pathlib.Path(mask_dir / "organoid_masks_reconstructed.tiff").resolve(
        strict=True
    )
    mask_output_path = pathlib.Path(
        mask_dir / "organoid_masks_reconstructed_corrected.tiff"
    ).resolve()

else:
    raise ValueError("Compartment must be either nuclei, cell or organoid")

mask = io.imread(mask_path)


# In[4]:


# # get the organoid masks
# organoid_mask_path = mask_dir / "organoid_masks_reconstructed.tiff"
# cell_mask_path = mask_dir / "cell_masks_reconstructed.tiff"
# mask_path = mask_dir / "masks_reconstructed.tiff"
# organoid_mask = io.imread(organoid_mask_path)
# cell_mask = io.imread(cell_mask_path)
# mask = io.imread(mask_path)


# In[5]:


# # output paths
# organoid_mask_output_path = mask_dir / "organoid_masks_reconstructed_corrected.tiff"
# cell_mask_output_path = mask_dir / "cell_masks_reconstructed_corrected.tiff"
# mask_output_path = mask_dir / "masks_reconstructed_corrected.tiff"


# In[6]:


# # get the first z slice of the image
# for z in range(cell_mask.shape[0]):
#     plt.subplot(1, 3, 1)
#     plt.imshow(organoid_mask[z, :, :], cmap="magma")
#     plt.title("Organoid mask")
#     plt.subplot(1, 3, 2)
#     plt.imshow(cell_mask[z, :, :], cmap="magma")
#     plt.title("Cell mask")
#     plt.subplot(1, 3, 3)
#     plt.imshow(mask[z, :, :], cmap="magma")
#     plt.title("Nuclei mask")
#     plt.show()


# In[7]:


def get_mask_properties(mask):
    list_of_cell_masks = []
    for z in range(mask.shape[0]):
        compartment_df = pd.DataFrame.from_dict(
            skimage.measure.regionprops_table(
                mask[z, :, :],
                properties=["centroid", "bbox"],
            )
        )
        compartment_df["z"] = z

        list_of_cell_masks.append(compartment_df)
    compartment_df = pd.concat(list_of_cell_masks)

    # get the pixel value of the organoid mask at each x,y,z coordinate
    compartment_df["label"] = mask[
        compartment_df["z"].astype(int),
        compartment_df["centroid-0"].astype(int),
        compartment_df["centroid-1"].astype(int),
    ]
    compartment_df.reset_index(drop=True, inplace=True)
    compartment_df["new_label"] = compartment_df["label"]
    # drop all labels that are 0
    compartment_df = compartment_df[compartment_df["label"] != 0]
    return compartment_df


# In[8]:


# list_of_cell_masks = []
# for z in range(mask.shape[0]):
#     compartment_df = pd.DataFrame.from_dict(
#         skimage.measure.regionprops_table(
#             mask[z, :, :],
#             properties=["centroid", "bbox"],
#         )
#     )
#     compartment_df["z"] = z

#     list_of_cell_masks.append(compartment_df)
# compartment_df = pd.concat(list_of_cell_masks)

# # get the pixel value of the organoid mask at each x,y,z coordinate
# compartment_df["label"] = mask[
#     compartment_df["z"].astype(int),
#     compartment_df["centroid-0"].astype(int),
#     compartment_df["centroid-1"].astype(int),
# ]
# compartment_df.reset_index(drop=True, inplace=True)


# In[ ]:


# In[9]:


from typing import Tuple


def calculate_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    Calculate the area of a bounding box.

    Parameters
    ----------
    bbox : Tuple[int, int, int, int]
        The bounding box coordinates in the format (x_min, y_min, x_max, y_max).

    Returns
    -------
    int
        The area of the bounding box.
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def calculate_overlap(
    bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
) -> float:
    # calculate the % overlap of the second bbox with the first bbox
    if calculate_bbox_area(bbox1) == 0 or calculate_bbox_area(bbox2) == 0:
        return 0.0
    if calculate_bbox_area(bbox1) >= calculate_bbox_area(bbox2):
        x_min = max(bbox1[0], bbox2[0])
        y_min = max(bbox1[1], bbox2[1])
        x_max = min(bbox1[2], bbox2[2])
        y_max = min(bbox1[3], bbox2[3])
        overlap_width = max(0, x_max - x_min)
        overlap_height = max(0, y_max - y_min)
        overlap_area = overlap_width * overlap_height
        bbox1_area = calculate_bbox_area(bbox1)
        bbox2_area = calculate_bbox_area(bbox2)
        overlap_percentage = overlap_area / bbox2_area if bbox2_area > 0 else 0
        return overlap_percentage
    elif calculate_bbox_area(bbox1) < calculate_bbox_area(bbox2):
        x_min = max(bbox1[0], bbox2[0])
        y_min = max(bbox1[1], bbox2[1])
        x_max = min(bbox1[2], bbox2[2])
        y_max = min(bbox1[3], bbox2[3])
        overlap_width = max(0, x_max - x_min)
        overlap_height = max(0, y_max - y_min)
        overlap_area = overlap_width * overlap_height
        bbox1_area = calculate_bbox_area(bbox1)
        bbox2_area = calculate_bbox_area(bbox2)
        overlap_percentage = overlap_area / bbox1_area if bbox1_area > 0 else 0
        return overlap_percentage
    else:
        raise ValueError("Invalid bounding box coordinates, check for NaN values")


def merge_sets(list_of_sets: list) -> list:
    for i, set1 in enumerate(list_of_sets):
        for j, set2 in enumerate(list_of_sets):
            if i != j and len(set1.intersection(set2)) > 0:
                set1.update(set2)
    return list_of_sets


# In[10]:


# first_index_df = compartment_df[compartment_df['label'] == 3].copy()
# rest_index_df = compartment_df[compartment_df['label'] != 3].copy()

# z_slices = [2,3,4]


# In[11]:


# interpolated_rows_to_add = []
# changes_catalog = {
#     "index": [],
#     "z": [],
#     "label": [],
#     "new_label": [],
# }


# In[12]:


sliding_window_context = 3
compartment_df = get_mask_properties(mask)
z_slices = compartment_df["z"].unique()
global_max_z = compartment_df["z"].max()
global_min_z = compartment_df["z"].min()
new_mask_image = mask.copy()
for z in z_slices[: -(sliding_window_context - 1)]:
    compartment_df = get_mask_properties(new_mask_image)
    interpolated_rows_to_add = []
    changes_catalog = {
        "index": [],
        "z": [],
        "label": [],
        "new_label": [],
    }
    # Get the temporary sliding window
    tmp_window_df = compartment_df[
        (compartment_df["z"] >= z) & (compartment_df["z"] < z + sliding_window_context)
    ]
    print(tmp_window_df["z"].unique())
    if tmp_window_df["z"].nunique() < sliding_window_context:
        pass
    tmp_window_df
    final_dict = {
        "index1": [],
        "index2": [],
        "z1": [],
        "z2": [],
        "bbox1": [],
        "bbox2": [],
        "overlap": [],
        "label1": [],
        "label2": [],
    }
    for i, row1 in tmp_window_df.iterrows():
        for j, row2 in tmp_window_df.iterrows():
            if i != j:  # Ensure you're not comparing the same row
                if row1["z"] != row2["z"]:
                    # get the first bbox
                    bbox1 = (
                        row1["bbox-0"],  # x-min
                        row1["bbox-1"],  # y-min
                        row1["bbox-2"],  # x-max
                        row1["bbox-3"],  # y-max
                    )
                    bbox2 = (
                        row2["bbox-0"],  # x-min
                        row2["bbox-1"],  # y-min
                        row2["bbox-2"],  # x-max
                        row2["bbox-3"],  # y-max
                    )
                    overlap = calculate_overlap(bbox1, bbox2)
                    if overlap > 0.5:
                        final_dict["index1"].append(i)
                        final_dict["index2"].append(j)
                        final_dict["z1"].append(row1["z"])
                        final_dict["z2"].append(row2["z"])
                        final_dict["bbox1"].append(bbox1)
                        final_dict["bbox2"].append(bbox2)
                        final_dict["overlap"].append(overlap)
                        final_dict["label1"].append(row1["label"])
                        final_dict["label2"].append(row2["label"])

    final_df = pd.DataFrame.from_dict(final_dict)
    final_df["index_set"] = final_df.apply(
        lambda row: frozenset([row["index1"], row["index2"]]), axis=1
    )
    final_df["index_set"] = final_df["index_set"].apply(lambda x: tuple(sorted(x)))
    final_df.drop_duplicates(subset=["index_set"], inplace=True)
    final_df
    list_of_sets = final_df["index_set"].tolist()
    list_of_sets = [set(s) for s in list_of_sets]
    merged_sets = merge_sets(list_of_sets)
    # drop the duplicates
    merged_sets = list({frozenset(s): s for s in merged_sets}.values())
    merged_sets

    # from final_df generate the z-ordered cases
    for object_set in merged_sets:
        # find rows that contain integers that are in the object_set
        rows = final_df[
            final_df["index_set"].apply(lambda x: set(x).issubset(object_set))
        ]
        # get the index, label, and z pair
        dict_of_object_information = {"index": [], "label": [], "z": []}
        for i, row in rows.iterrows():
            dict_of_object_information["index"].append(row["index1"])
            dict_of_object_information["label"].append(row["label1"])
            dict_of_object_information["z"].append(row["z1"])
            dict_of_object_information["index"].append(row["index2"])
            dict_of_object_information["label"].append(row["label2"])
            dict_of_object_information["z"].append(row["z2"])
        dict_of_object_information_df = pd.DataFrame.from_dict(
            dict_of_object_information
        )
        dict_of_object_information_df.drop_duplicates(
            subset=["index", "label", "z"], inplace=True
        )
        dict_of_object_information_df.sort_values(by=["index", "z"], inplace=True)
        object_max_slice_label = dict_of_object_information_df.loc[
            dict_of_object_information_df["z"]
            == dict_of_object_information_df["z"].max(),
            "label",
        ].values
        object_min_slice_label = dict_of_object_information_df.loc[
            dict_of_object_information_df["z"]
            == dict_of_object_information_df["z"].min(),
            "label",
        ].values
        max_z = dict_of_object_information_df["z"].max()
        min_z = dict_of_object_information_df["z"].min()
        # check that the min and max slice labels are only one
        if len(object_max_slice_label) > 1:
            # print("Error: More than one min slice label")
            # print(object_min_slice_label)
            object_max_slice_label = object_max_slice_label[0].item()
        else:
            object_max_slice_label = object_max_slice_label[0]
        if len(object_min_slice_label) > 1:
            # print("Error: More than one min slice label")
            # print(object_min_slice_label)
            # select the first one
            object_min_slice_label = object_min_slice_label[0].item()
        else:
            object_min_slice_label = object_min_slice_label[0]
        if object_max_slice_label == object_min_slice_label:
            # print("The min and max slice labels are the same")
            pass
        else:
            # print("The min and max slice labels are different")
            pass

        # check if the min slice and max slice labels are the same:

        ## logic that checks for a sandwich case
        for i, row in dict_of_object_information_df.iterrows():
            if (
                row["z"] != dict_of_object_information_df["z"].max()
                and row["z"] != dict_of_object_information_df["z"].min()
            ):
                if object_max_slice_label == object_min_slice_label:
                    if (
                        row["label"] != object_max_slice_label
                        and row["label"] != object_min_slice_label
                    ):
                        # print(row)
                        # print(row['index'])
                        changes_catalog["index"].append(row["index"])
                        changes_catalog["z"].append(row["z"])
                        changes_catalog["label"].append(row["label"])
                        changes_catalog["new_label"].append(object_max_slice_label)
        # logic that checks for a missing slice mask
        if len(dict_of_object_information_df["label"].unique()) < 3:
            if object_max_slice_label == object_min_slice_label:
                middle_slice = int((max_z + min_z) / 2)
                # check for the presence of the middle slice
                if max_z - min_z > 1:
                    # print("missing object detected")
                    # interpolate the labels to the middle most slice
                    # get the middle slice
                    middle_slice = int((max_z + min_z) / 2)
                    # insert one slice
                    if row["z"] == global_max_z:
                        z_zlice_to_copy = row["z"] - 1
                    else:
                        z_zlice_to_copy = row["z"] + 1
                    new_row = {
                        # 'index': dict_of_object_information_df['index'].values[0],
                        # 'index': object_max_slice_label,
                        "added_z": middle_slice,
                        "added_new_label": object_max_slice_label,
                        "zslice_to_copy": z_zlice_to_copy,
                    }
                    interpolated_rows_to_add.append(pd.DataFrame(new_row, index=[0]))
        # find labels that are 1 slice away from the min or max and extend the label
        for i, row in dict_of_object_information_df.iterrows():
            # check if the z slice is one away from the min or max (global min and max)
            if row["z"] == global_max_z - 1:
                new_row = {
                    # 'index': dict_of_object_information_df['index'].values[0],
                    # 'index': object_max_slice_label,
                    "added_z": global_max_z,
                    "added_new_label": row["label"],
                    "zslice_to_copy": row["z"],
                }
                interpolated_rows_to_add.append(pd.DataFrame(new_row, index=[0]))
            elif row["z"] == global_min_z + 1:
                new_row = {
                    # 'index': dict_of_object_information_df['index'].values[0],
                    # 'index': object_max_slice_label,
                    "added_z": global_min_z,
                    "added_new_label": row["label"],
                    "zslice_to_copy": row["z"],
                }
                interpolated_rows_to_add.append(pd.DataFrame(new_row, index=[0]))

    # edit the changes on the image
    changes_catalog_df = pd.DataFrame.from_dict(changes_catalog)

    for i, row in changes_catalog_df.iterrows():
        compartment_df.loc[
            (compartment_df["z"] == row["z"])
            & (compartment_df["label"] == row["label"]),
            "new_label",
        ] = row["new_label"]

    # mask label reassignment
    for slice in range(new_mask_image.shape[0]):
        tmp_mask = new_mask_image[slice, :, :]
        tmp_df = compartment_df[compartment_df["z"] == slice]
        for i in range(tmp_df.shape[0]):
            tmp_mask[tmp_mask == tmp_df.iloc[i]["label"]] = tmp_df.iloc[i]["new_label"]

        new_mask_image[slice, :, :] = tmp_mask
    if len(interpolated_rows_to_add) > 0:
        interpolated_rows_to_add_df = pd.concat(interpolated_rows_to_add, axis=0)

    for slice in range(new_mask_image.shape[0]):
        tmp_mask = new_mask_image[slice, :, :]
        tmp_df = compartment_df[compartment_df["z"] == slice]
        tmp_interpolated_rows_to_add_df = interpolated_rows_to_add_df[
            interpolated_rows_to_add_df["added_z"] == slice
        ]
        for i, row in tmp_interpolated_rows_to_add_df.iterrows():
            # get the z slice to copy mask
            mask_to_copy = new_mask_image[
                row["zslice_to_copy"].astype(int), :, :
            ].copy()
            mask_to_copy_original = mask_to_copy.copy()
            mask_to_copy[mask_to_copy != row["added_new_label"]] = 0

            new_slice = new_mask_image[row["added_z"].astype(int), :, :].copy()
            max_projected_slice = np.maximum(new_slice, mask_to_copy)
            new_mask_image[row["added_z"].astype(int), :, :] = max_projected_slice


# In[13]:


tifffile.imwrite(mask_output_path, new_mask_image)


# In[14]:


# changes_catalog_df = pd.DataFrame.from_dict(changes_catalog)
# compartment_df_before = compartment_df.copy()
# compartment_df_before.head()
# for i, row in changes_catalog_df.iterrows():
#     # replace the new label
#     pass
#     compartment_df.loc[
#         (compartment_df["z"] == row["z"]) & (compartment_df["label"] == row["label"]), "new_label"
#     ] = row["new_label"]
# print(compartment_df_before.shape, compartment_df.shape)
# # sanity check that we actually changed the labels
# checking_df = pd.concat([compartment_df, compartment_df_before], axis=0)
# checking_df.drop_duplicates(inplace=True)
# checking_df

# new_mask_image = np.zeros_like(mask)
# # mask label reassignment
# for slice in range(mask.shape[0]):
#     tmp_mask = mask[slice, :, :]
#     tmp_df = compartment_df[compartment_df["z"] == slice]
#     for i in range(tmp_df.shape[0]):
#         tmp_mask[tmp_mask == tmp_df.iloc[i]["label"]] = tmp_df.iloc[i]["new_label"]

#     new_mask_image[slice, :, :] = tmp_mask
# interpolated_rows_to_add_df = pd.concat(interpolated_rows_to_add, axis=0)
# print(interpolated_rows_to_add_df.shape)
# interpolated_rows_to_add_df
# for slice in range(new_mask_image.shape[0]):
#     tmp_mask = mask[slice, :, :]
#     tmp_df = compartment_df[compartment_df["z"] == slice]
#     tmp_interpolated_rows_to_add_df = interpolated_rows_to_add_df[
#         interpolated_rows_to_add_df["added_z"] == slice
#     ]
#     for i, row in tmp_interpolated_rows_to_add_df.iterrows():
#         # get the z slice to copy mask
#         mask_to_copy = new_mask_image[row["zslice_to_copy"].astype(int), :, :].copy()
#         mask_to_copy_original = mask_to_copy.copy()
#         mask_to_copy[mask_to_copy != row["added_new_label"]] = 0

#         new_slice = new_mask_image[row["added_z"].astype(int), :, :].copy()
#         max_projected_slice = np.maximum(new_slice, mask_to_copy)
#         new_mask_image[row["added_z"].astype(int), :, :] = max_projected_slice


# In[ ]:
