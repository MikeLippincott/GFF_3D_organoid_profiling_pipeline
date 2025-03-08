import pathlib
from typing import Union

import arrow
import numpy
import pandas


def path_checking(path: pathlib.Path) -> Union[ValueError, None]:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if path is None:
        raise ValueError("Output path must be a non-None value")
    if not path.suffix == ".parquet":
        raise ValueError("Output path must be a parquet file")
    return None


def organize_featurization_data(
    features: dict,
    compartment: str,
    channel: str,
    label_index: int,
    image_set_name: str,
    return_df: bool = True,
    output: bool = False,
    output_path: pathlib.Path = None,
) -> Union[pathlib.Path or pandas.DataFrame]:
    updated_features = {}
    for key, value in features.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                updated_features[f"{compartment}_{channel}_{key}_{sub_key}"] = sub_value
        else:
            updated_features[f"{compartment}_{channel}_{key}_nosubfeature"] = value
    # make a df out of the features
    df = pandas.DataFrame(updated_features, index=[0])
    # get the index of the object being measured
    df.insert(0, "object_index", label_index)
    df.insert(0, "image_set", image_set_name)
    if output:
        path_checking(output_path)
        df.to_parquet(output_path)
    if return_df:
        return df
