#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import pathlib
import sys

import pandas as pd
from cytotable import convert, presets

sys.path.append("../../../utils")
import uuid

from parsl.config import Config
from parsl.executors import HighThroughputExecutor

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[6]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID to process, e.g. 'P01'",
    )
    argparser.add_argument(
        "--well_fov",
        type=str,
        required=True,
        help="Well and field of view to process, e.g. 'A01_1'",
    )
    args = argparser.parse_args()
    patient = args.patient
    well_fov = args.well_fov
else:
    patient = "NF0014"
    well_fov = "C4-2"


# In[ ]:


input_sqlite_file = pathlib.Path(
    f"../../data/{patient}/converted_profiles/{well_fov}.sqlite"
).resolve(strict=True)
destination_sc_parquet_file = pathlib.Path(
    f"../../data/{patient}/converted_profiles/{well_fov}.parquet"
).resolve()
destination_organoid_parquet_file = pathlib.Path(
    f"../../data/{patient}/converted_profiles/{well_fov}.parquet"
).resolve()
dest_datatype = "parquet"


# In[ ]:


# preset configurations based on typical CellProfiler outputs
preset = "cellprofiler_sqlite_pycytominer"
presets.config[preset][
    "CONFIG_JOINS"
    # remove Image_Metadata_Plate from SELECT as this metadata was not extracted from file names
    # add Image_Metadata_FOV as this is an important metadata when finding where single cells are located
] = """WITH Per_Image_Filtered AS (
            SELECT
                *
            FROM
                Per_Cytoplasm AS per_cytoplasm
            LEFT JOIN read_parquet('per_cells.parquet') AS per_cells ON
                per_cells.object_id = per_cytoplasm.object_id
            LEFT JOIN read_parquet('per_nuclei.parquet') AS per_nuclei ON
                per_nuclei.object_id = per_cytoplasm.object_id
                AND per_nuclei.object_id = per_cytoplasm.object_id
            """


# In[ ]:


# merge single cells and output as parquet file
convert(
    source_path=input_sqlite_file,
    dest_path=destination_sc_parquet_file,
    dest_datatype=dest_datatype,
    preset=preset,
    parsl_config=Config(
        executors=[HighThroughputExecutor()],
        run_dir=f"run_dir_{uuid.uuid4().hex}",
    ),
    chunk_size=1000,
)


# ### Get the organoid profile

# In[ ]:


# preset configurations based on typical CellProfiler outputs
preset = "cellprofiler_sqlite_pycytominer"
presets.config[preset][
    "CONFIG_JOINS"
    # remove Image_Metadata_Plate from SELECT as this metadata was not extracted from file names
    # add Image_Metadata_FOV as this is an important metadata when finding where single cells are located
] = """WITH Per_Image_Filtered AS (
            SELECT
                *
"""


# In[ ]:


# merge single cells and output as parquet file
convert(
    source_path=input_sqlite_file,
    dest_path=destination_organoid_parquet_file,
    dest_datatype=dest_datatype,
    preset=preset,
    parsl_config=Config(
        executors=[HighThroughputExecutor()],
        run_dir=f"run_dir_{uuid.uuid4().hex}",
    ),
    chunk_size=1000,
)
