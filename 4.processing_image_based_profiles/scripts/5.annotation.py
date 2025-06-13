#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile annotation.
# The platemap is mapped back to the profile to retain the sample metadata.
#

# In[1]:


import argparse
import pathlib

import pandas as pd

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID to process, e.g. 'P01'",
    )
    args = argparser.parse_args()
    patient = args.patient

else:
    patient = "NF0014"


# In[3]:


def annotate_profiles(
    profile_df: pd.DataFrame, platemap_df: pd.DataFrame, patient: str
) -> pd.DataFrame:
    """
    Annotate profiles with treatment, dose, and unit information from the platemap.

        Parameters
        ----------
        profile_df : pd.DataFrame
            Profile DataFrame containing image_set information.
            Could be either single-cell or organoid profiles.
        platemap_df : pd.DataFrame
            Platmap DataFrame containing well_position, treatment, dose, and unit.
        patient : str
            Patient ID to annotate the profiles with.

        Returns
        -------
        pd.DataFrame
            Annotated profile DataFrame with additional columns for treatment, dose, and unit.
    """
    profile_df["Well"] = profile_df["image_set"].str.split("-").str[0]
    profile_df.insert(2, "Well", profile_df.pop("Well"))
    profile_df = pd.merge(
        profile_df,
        platemap_df[["well_position", "treatment", "dose", "unit"]],
        left_on="Well",
        right_on="well_position",
        how="left",
    ).drop(columns=["well_position"])
    for col in ["treatment", "dose", "unit"]:
        profile_df.insert(1, col, profile_df.pop(col))
    profile_df.insert(0, "patient", patient)
    return profile_df


# In[4]:


# pathing
sc_merged_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/0.sc_merged_profiles.parquet"
).resolve(strict=True)
organoid_merged_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/0.organoid_merged_profiles.parquet"
).resolve(strict=True)

platemap_path = pathlib.Path("../../data/NF0014/platemap/platemap.csv").resolve(
    strict=True
)

# output path
sc_annotated_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/1.sc_annotated_profiles.parquet"
).resolve()
organoid_annotated_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/1.organoid_annotated_profiles.parquet"
).resolve()


# In[5]:


# read data
sc_merged = pd.read_parquet(sc_merged_path)
organoid_merged = pd.read_parquet(organoid_merged_path)
# read platemap
platemap = pd.read_csv(platemap_path)


# In[6]:


sc_merged = annotate_profiles(sc_merged, platemap, patient)
organoid_merged = annotate_profiles(organoid_merged, platemap, patient)


# In[7]:


sc_merged.head()


# In[8]:


organoid_merged.head()


# In[9]:


# save annotated profiles
sc_merged.to_parquet(sc_annotated_output_path, index=False)
organoid_merged.to_parquet(organoid_annotated_output_path, index=False)
