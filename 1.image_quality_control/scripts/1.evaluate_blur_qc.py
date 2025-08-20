#!/usr/bin/env python
# coding: utf-8

# # Whole image quality control metric evaluation - Blur
#
# In this notebook, we will use the outputted QC metrics per image (every z-slice per channel) to start working on developing thresholds using z-score to flag images during CellProfiler processing.
# We are loading in the results from the preliminary data (across three patients) to attempt to develop generalizable thresholds.
# This data is 3D, so we are decide if it make sense to remove a whole organoid based on if one z-slice fails.
#
# ## Blurry image detection
#
# For detecting poor quality images based on blur, we use the feature `PowerLogLogSlope`, where more negative values indicate blurry images.
# We first create distribution plots per plates and per channel to evaluate if the distributions across channels are different.
# We will use this to determine if we process the data with all channels combined or separately.
#
# We will use a method called `coSMicQC`, which takes a feature of interest and detect outliers based on z-scoring and how far from the mean that outliers will be.

# ## Import libraries

# In[ ]:


import pathlib
import sys

import cosmicqc
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Get the current working directory
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
from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()


# ## Set paths and variables

# In[ ]:


# Base metadata columns that are constant across channels
base_metadata_columns = [
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_Zslice",
    "Metadata_zslice_total",
]

# Directory for figures to be outputted
figure_dir = pathlib.Path(f"{root_dir}/1.image_quality_control/qc_figures")
figure_dir.mkdir(exist_ok=True)

# Directory containing the QC results
qc_results_dir = pathlib.Path(f"{root_dir}/1.image_quality_control/qc_results")

# Directory for output file with flags
qc_output_dir = pathlib.Path(f"{root_dir}/1.image_quality_control/qc_flag_files")
# Ensure the QC results directory exists
qc_output_dir.mkdir(exist_ok=True)

# Find all Image.csv files for all plates using glob
image_csv_paths = qc_results_dir.glob("*/Image.csv")

# Path to the template pipeline file to update with proper thresholds for flagging
pipeline_path = pathlib.Path(
    f"{root_dir}/1.image_quality_control/pipeline/template_flag_pipeline.cppipe"
)


# ## Load in QC results per plate and combine

# In[ ]:


# Define prefixes for columns to select
prefixes = (
    "Metadata",
    "FileName",
    "PathName",
    "ImageQuality_PowerLogLogSlope",
)

# Load and concatenate the data for all plates
qc_dfs = []
for path in image_csv_paths:
    # Load only the required columns by filtering columns with specified prefixes
    plate_df = pd.read_csv(path, usecols=lambda col: col.startswith(prefixes))

    # Skip processing NF0017
    if "NF0017" in str(path):
        continue

    # Check for NaNs in the Metadata_Plate, Metadata_Well, and Metadata_Site columns
    if (
        plate_df[["Metadata_Plate", "Metadata_Well", "Metadata_Site"]]
        .isna()
        .any()
        .any()
    ):
        print(
            f"NaNs detected in {path} in Metadata_Plate, Metadata_Well, or Metadata_Site columns"
        )

    # Fill NaNs for specific conditions
    if "NF0018_T6_qc_results" in str(path):
        plate_df["Metadata_Plate"] = plate_df["Metadata_Plate"].fillna("NF0018_T6")
        plate_df["Metadata_Well"] = plate_df["Metadata_Well"].fillna("E5")
        plate_df["Metadata_Site"] = plate_df["Metadata_Site"].fillna(3)

    qc_dfs.append(plate_df)


# In[4]:


# Concatenate all plate data into a single dataframe
concat_qc_df = pd.concat(qc_dfs, ignore_index=True)

# Add a new column Metadata_zslice_total which gets the number of rows per Metadata_Plate, Metadata_Well, and Metadata_Site unique combos
concat_qc_df["Metadata_zslice_total"] = concat_qc_df.groupby(
    ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
)["Metadata_Site"].transform("count")

print(concat_qc_df.shape)
concat_qc_df.head(2)


# ## Generate blur density distribution plot

# In[5]:


# Step 1: Select only the columns containing "PowerLogLogSlope" and keep Metadata_Plate
relevant_columns = [col for col in concat_qc_df.columns if "PowerLogLogSlope" in col]
relevant_columns.insert(0, "Metadata_Plate")

# Filter the dataframe
filtered_df = concat_qc_df[relevant_columns]

# Step 2: Reshape the dataframe into a long format
long_df = filtered_df.melt(
    id_vars=["Metadata_Plate"], var_name="Channel", value_name="PowerLogLogSlope"
)

# Step 3: Clean up the channel names
long_df["Channel"] = long_df["Channel"].str.replace(
    "ImageQuality_PowerLogLogSlope_", ""
)

# Drop rows with missing or non-numeric values
long_df = long_df.dropna(subset=["PowerLogLogSlope"])

# Define the custom color palette
custom_palette = {
    "DNA": "blue",
    "ER": "green",
    "AGP": "red",
    "Mito": "purple",
    "Brightfield": "orange",
}

# Step 4: Create the facet grid
g = sns.displot(
    data=long_df,
    x="PowerLogLogSlope",
    hue="Channel",
    palette=custom_palette,  # Use the custom palette
    col="Metadata_Plate",
    kind="kde",
    fill=True,
    alpha=0.5,
    col_wrap=3,  # Adjust for 3 columns per row
)

# Customize the plot
g.set_titles("Plate: {col_name}")
g.set_axis_labels("PowerLogLogSlope", "Density")
g.tight_layout()

# Save the plot
plt.savefig(figure_dir / "blur_density_distribution.png", dpi=500)

# Show the plot
plt.show()


# Given the plot, we can conclude that the distributions across channels are very different.
# We will process each channel independently and determine thresholds per channel.
# The distributions across plate look similar, so we will not be determining per plate.

# ## Detect blur in DNA channel

# In[6]:


channel = "DNA"

# Add the dynamic PathName and FileName for the channel
metadata_columns = base_metadata_columns + [
    f"PathName_{channel}",
    f"FileName_{channel}",
]


# Find large nuclei outliers for the current plate
blur_DNA_outliers = cosmicqc.find_outliers(
    df=concat_qc_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_DNA": -1.75,
    },
)

pd.DataFrame(blur_DNA_outliers)


# In[7]:


# Combine PathName and FileName columns to construct full paths for DNA
blur_DNA_outliers["Full_Path_DNA"] = (
    blur_DNA_outliers["PathName_DNA"] + "/" + blur_DNA_outliers["FileName_DNA"]
)

# Create a figure to display images
plt.figure(figsize=(15, 5))

# Loop through the first 3 rows of the blur_DNA_outliers dataframe and display each image
for idx, row in enumerate(blur_DNA_outliers.itertuples(), start=1):
    if idx > 3:  # Only display the first 3 images
        break
    image_path = row.Full_Path_DNA
    # Format the metadata title
    metadata_title = f"{row.Metadata_Plate}_{row.Metadata_Well}-{int(row.Metadata_Site)}_{row.Metadata_Zslice} (Total z-slices: {row.Metadata_zslice_total})"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        continue
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for proper display

    # Add the image to the plot
    plt.subplot(1, 3, idx)  # Use idx for subplot placement
    plt.imshow(image)
    plt.title(metadata_title)  # Set the formatted metadata as the title
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()


# ## Detect blur in Mito channel

# In[8]:


channel = "Mito"

# Add the dynamic PathName and FileName for the channel
metadata_columns = base_metadata_columns + [
    f"PathName_{channel}",
    f"FileName_{channel}",
]

# Find large nuclei outliers for the current plate
blur_Mito_outliers = cosmicqc.find_outliers(
    df=concat_qc_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_Mito": -2,
    },
)

pd.DataFrame(blur_Mito_outliers)

blur_Mito_outliers = blur_Mito_outliers.sort_values(
    by="ImageQuality_PowerLogLogSlope_Mito", ascending=True
)

blur_Mito_outliers.head()


# In[9]:


# Combine PathName and FileName columns to construct full paths for Mito
blur_Mito_outliers["Full_Path_Mito"] = (
    blur_Mito_outliers["PathName_Mito"] + "/" + blur_Mito_outliers["FileName_Mito"]
)

# Group by Plate, Well, and Site to ensure uniqueness
unique_groups = blur_Mito_outliers.groupby(
    ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
)
print(len(unique_groups))

# Randomly sample one row per group
unique_samples = unique_groups.apply(lambda group: group.sample(n=1, random_state=0))

# Reset the index for convenience
unique_samples = unique_samples.reset_index(drop=True)

# Further randomly select 3 unique images
if len(unique_samples) < 3:
    print("Not enough unique Plate-Well-Site combinations for the requested images.")
else:
    selected_images = unique_samples.sample(n=3, random_state=0)

# Create a figure to display images
plt.figure(figsize=(15, 5))

# Loop through the selected image paths and display each image
for idx, row in enumerate(selected_images.itertuples(), start=1):
    image_path = row.Full_Path_Mito
    # Format the metadata title
    metadata_title = f"{row.Metadata_Plate}_{row.Metadata_Well}-{int(row.Metadata_Site)}_{row.Metadata_Zslice} (Total z-slices: {row.Metadata_zslice_total})"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        continue
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for proper display

    # Add the image to the plot
    plt.subplot(1, 3, idx)  # Use idx for subplot placement
    plt.imshow(image)
    plt.title(metadata_title)  # Set the formatted metadata as the title
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()


# ## Detect blur in ER channel

# In[10]:


channel = "ER"

# Add the dynamic PathName and FileName for the channel
metadata_columns = base_metadata_columns + [
    f"PathName_{channel}",
    f"FileName_{channel}",
]

# Find large nuclei outliers for the current plate
blur_er_outliers = cosmicqc.find_outliers(
    df=concat_qc_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_ER": -2,
    },
)

pd.DataFrame(blur_er_outliers).head()


# In[11]:


# Combine PathName and FileName columns to construct full paths
blur_er_outliers["Full_Path_ER"] = (
    blur_er_outliers["PathName_ER"] + "/" + blur_er_outliers["FileName_ER"]
)

# Group by Plate, Well, and Site to ensure uniqueness
unique_groups = blur_er_outliers.groupby(
    ["Metadata_Plate", "Metadata_Well", "Metadata_Site"],
)
print(len(unique_groups))

# Randomly sample one row per group
unique_samples = unique_groups.apply(lambda group: group.sample(n=1, random_state=0))

# Reset the index for convenience
unique_samples = unique_samples.reset_index(drop=True)

# Further randomly select 3 unique images
if len(unique_samples) < 3:
    print("Not enough unique Plate-Well-Site combinations for the requested images.")
else:
    selected_images = unique_samples.sample(n=3, random_state=0)

# Create a figure to display images
plt.figure(figsize=(15, 5))

# Loop through the selected image paths and display each image
for idx, row in enumerate(
    selected_images.itertuples(), start=1
):  # Enumerate for subplot indexing
    image_path = row.Full_Path_ER
    # Format the metadata title
    metadata_title = f"{row.Metadata_Plate}_{row.Metadata_Well}-{int(row.Metadata_Site)}_{row.Metadata_Zslice} (Total z-slices: {row.Metadata_zslice_total})"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        continue
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for proper display

    # Add the image to the plot
    plt.subplot(1, 3, idx)  # Use idx for subplot placement
    plt.imshow(image)
    plt.title(metadata_title)
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()


# ## Detect blur in AGP channel

# In[12]:


channel = "AGP"

# Add the dynamic PathName and FileName for the channel
metadata_columns = base_metadata_columns + [
    f"PathName_{channel}",
    f"FileName_{channel}",
]

# Find large nuclei outliers for the current plate
blur_agp_outliers = cosmicqc.find_outliers(
    df=concat_qc_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_AGP": -2.25,
    },
)

pd.DataFrame(blur_agp_outliers).head()


# In[13]:


# Combine PathName and FileName columns to construct full paths
blur_agp_outliers["Full_Path_AGP"] = (
    blur_agp_outliers["PathName_AGP"] + "/" + blur_agp_outliers["FileName_AGP"]
)

# Group by Plate, Well, and Site to ensure uniqueness
unique_groups = blur_agp_outliers.groupby(
    ["Metadata_Plate", "Metadata_Well", "Metadata_Site"],
)
print(len(unique_groups))

# Randomly sample one row per group
unique_samples = unique_groups.apply(lambda group: group.sample(n=1, random_state=0))

# Reset the index for convenience
unique_samples = unique_samples.reset_index(drop=True)

# Further randomly select 3 unique images
if len(unique_samples) < 3:
    print("Not enough unique Plate-Well-Site combinations for the requested images.")
else:
    selected_images = unique_samples.sample(n=3, random_state=0)

# Create a figure to display images
plt.figure(figsize=(15, 5))

# Loop through the selected image paths and display each image
for idx, row in enumerate(
    selected_images.itertuples(), start=1
):  # Enumerate for subplot indexing
    image_path = row.Full_Path_AGP
    # Format the metadata title
    metadata_title = f"{row.Metadata_Plate}_{row.Metadata_Well}-{int(row.Metadata_Site)}_{row.Metadata_Zslice} (Total z-slices: {row.Metadata_zslice_total})"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        continue
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for proper display

    # Add the image to the plot
    plt.subplot(1, 3, idx)  # Use idx for subplot placement
    plt.imshow(image)
    plt.title(metadata_title)
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()


# ## Detect blur in Brightfield channel

# In[14]:


channel = "Brightfield"

# Add the dynamic PathName and FileName for the channel
metadata_columns = base_metadata_columns + [
    f"PathName_{channel}",
    f"FileName_{channel}",
]

# Find large nuclei outliers for the current plate
blur_brightfield_outliers = cosmicqc.find_outliers(
    df=concat_qc_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_Brightfield": -3,
    },
)

pd.DataFrame(blur_brightfield_outliers).head()


# In[15]:


# Combine PathName and FileName columns to construct full paths
blur_brightfield_outliers["Full_Path_Brightfield"] = (
    blur_brightfield_outliers["PathName_Brightfield"]
    + "/"
    + blur_brightfield_outliers["FileName_Brightfield"]
)

# Group by Plate, Well, and Site to ensure uniqueness
unique_groups = blur_brightfield_outliers.groupby(
    ["Metadata_Plate", "Metadata_Well", "Metadata_Site"],
)
print(len(unique_groups))

# Randomly sample one row per group (select random state for best examples)
unique_samples = unique_groups.apply(lambda group: group.sample(n=1, random_state=0))

# Reset the index for convenience
unique_samples = unique_samples.reset_index(drop=True)

# Further randomly select 3 unique images (select random state for best examples)
if len(unique_samples) < 3:
    print("Not enough unique Plate-Well-Site combinations for the requested images.")
else:
    selected_images = unique_samples.sample(n=3, random_state=0)

# Create a figure to display images
plt.figure(figsize=(15, 5))

# Loop through the selected image paths and display each image
for idx, row in enumerate(
    selected_images.itertuples(), start=1
):  # Enumerate for subplot indexing
    image_path = row.Full_Path_Brightfield
    # Format the metadata title
    metadata_title = f"{row.Metadata_Plate}_{row.Metadata_Well}-{int(row.Metadata_Site)}_{row.Metadata_Zslice} (Total z-slices: {row.Metadata_zslice_total})"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        continue
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for proper display

    # Add the image to the plot
    plt.subplot(1, 3, idx)  # Use idx for subplot placement
    plt.imshow(image)
    plt.title(metadata_title)
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()


# ## Create parquet file with each plate/well/site combos and boolean for pass/fail blur per channel

# In[16]:


# Combine all blur outliers dataframes into a single dataframe
blur_outliers = pd.concat(
    [
        blur_DNA_outliers,
        blur_Mito_outliers,
        blur_agp_outliers,
        blur_brightfield_outliers,
        blur_er_outliers,
    ],
    keys=["DNA", "Mito", "AGP", "Brightfield", "ER"],
    names=["Channel"],
).reset_index(level="Channel")

# Create a new dataframe with only Metadata_Plate, Metadata_Well, Metadata_Site, and Metadata_Zslice columns to assign blur flags
blur_outliers_per_zslice = concat_qc_df[
    ["Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_Zslice"]
].drop_duplicates()

# Initialize columns for each channel with False
for channel in ["DNA", "Mito", "AGP", "Brightfield", "ER"]:
    blur_outliers_per_zslice[f"Blurry_{channel}"] = False

# Flag the combos for blur detection
for channel in ["DNA", "Mito", "AGP", "Brightfield", "ER"]:
    blur_combos = blur_outliers[blur_outliers["Channel"] == channel][
        ["Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_Zslice"]
    ].drop_duplicates()
    blur_outliers_per_zslice.loc[
        blur_outliers_per_zslice.set_index(
            ["Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_Zslice"]
        ).index.isin(
            blur_combos.set_index(
                ["Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_Zslice"]
            ).index
        ),
        f"Blurry_{channel}",
    ] = True

# Reset the index on the unique combos dataframe
blur_outliers_per_zslice = blur_outliers_per_zslice.reset_index(drop=True)

# Save the blur_outliers_per_zslice dataframe to a parquet file
blur_outliers_per_zslice.to_parquet(qc_output_dir / "all_plates_qc_results.parquet")

# Print the number of rows with at least one Blurry column set to True
num_blurry_rows = (
    blur_outliers_per_zslice.loc[:, "Blurry_DNA":"Blurry_ER"].any(axis=1).sum()
)
print(
    f"Number of z-slices across all organoids detected as poor quality due to blur (in any channel): {num_blurry_rows}"
)

# Calculate and print the percentage of organoids detected as containing blur
percentage_blurry = (num_blurry_rows / len(blur_outliers_per_zslice)) * 100
print(
    f"Percentage of z-slices detected as poor quality due to blur: {percentage_blurry:.2f}%"
)

# Display the resulting dataframe
print(blur_outliers_per_zslice.shape)
blur_outliers_per_zslice.head()
