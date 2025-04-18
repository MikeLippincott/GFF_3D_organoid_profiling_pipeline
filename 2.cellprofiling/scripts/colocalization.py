#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import sys
import time

sys.path.append("../featurization_utils")
import itertools
import multiprocessing
from functools import partial
from itertools import product

import pandas as pd
from colocalization_utils import (
    measure_3D_colocalization,
    prepare_two_images_for_colocalization,
)
from loading_classes import ImageSetLoader, TwoObjectLoader

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[ ]:


def process_combination(args, image_set_loader):
    """
    Process a single combination of compartment and channel pair for colocalization analysis.

    Parameters
    ----------
    args : tuple
        A tuple containing the compartment, channel1, and channel2.
        Yes, order matters.
        args = (compartment, channel1, channel2)
        Where:
        compartment : str
            The compartment to process.
        channel1 : str
            The first channel to process.
        channel2 : str
            The second channel to process.

    image_set_loader : ImageSetLoader
        An instance of the ImageSetLoader class that loads the images and metadata.

    Returns
    -------
    str
        A message indicating the completion of processing for the given combination.
    """
    compartment, channel1, channel2 = args
    coloc_loader = TwoObjectLoader(
        image_set_loader=image_set_loader,
        compartment=compartment,
        channel1=channel1,
        channel2=channel2,
    )
    list_of_dfs = []
    for object_id in coloc_loader.object_ids:
        cropped_image1, cropped_image2 = prepare_two_images_for_colocalization(
            label_object1=coloc_loader.label_image,
            label_object2=coloc_loader.label_image,
            image_object1=coloc_loader.image1,
            image_object2=coloc_loader.image2,
            object_id1=object_id,
            object_id2=object_id,
        )
        colocalization_features = measure_3D_colocalization(
            cropped_image_1=cropped_image1,
            cropped_image_2=cropped_image2,
            thr=15,
            fast_costes="Accurate",
        )
        coloc_df = pd.DataFrame(colocalization_features, index=[0])
        coloc_df.columns = [
            f"Colocalization_{compartment}_{channel1}.{channel2}_{col}"
            for col in coloc_df.columns
        ]
        coloc_df.insert(0, "object_id", object_id)
        coloc_df.insert(1, "image_set", image_set_loader.image_set_name)
        list_of_dfs.append(coloc_df)

    coloc_df = pd.concat(list_of_dfs, ignore_index=True)
    output_file = pathlib.Path(
        f"../results/{image_set_loader.image_set_name}/Colocalization_{compartment}_{channel1}.{channel2}_features.parquet"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    coloc_df.to_parquet(output_file)

    return f"Processed {compartment} - {channel1}.{channel2}"


# In[2]:


image_set_path = pathlib.Path("../../data/NF0014/cellprofiler/C4-2/")


# In[3]:


channel_mapping = {
    "DNA": "405",
    "AGP": "488",
    "ER": "555",
    "Mito": "640",
    "BF": "TRANS",
    "Nuclei": "nuclei_",
    "Cell": "cell_",
    "Cytoplasm": "cytoplasm_",
    "Organoid": "organoid_",
}


# In[4]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[5]:


# get all channel combinations
channel_combinations = list(itertools.combinations(image_set_loader.image_names, 2))


# In[6]:


start_time = time.time()


# runs upon converted script execution

# In[ ]:


if __name__ == "__main__":
    # Generate all combinations of compartments and channel pairs
    combinations = list(
        product(
            image_set_loader.compartments,
            [pair for pair in channel_combinations],
        )
    )

    # Flatten the channel combinations for easier unpacking
    combinations = [
        (compartment, channel1, channel2)
        for compartment, (channel1, channel2) in combinations
    ]

    # Specify the number of cores to use
    cores_to_use = multiprocessing.cpu_count()  # Adjust the number of cores as needed
    print(f"Using {cores_to_use} cores for processing.")

    # Use multiprocessing to process combinations in parallel
    with multiprocessing.Pool(processes=cores_to_use) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(process_combination, image_set_loader=image_set_loader),
                    combinations,
                ),
                desc="Processing combinations",
            )
        )

    print("Processing complete.")


# In[ ]:


print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
