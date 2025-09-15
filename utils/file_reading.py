import numpy as np
import tifffile


def read_zstack_image(file_path: str) -> np.ndarray:
    """
    Description
    -----------
    Reads in a z-stack image from a given file path and returns it as a numpy array.

    Parameters
    ----------
    file_path : str
        The path to the z-stack image file.
    Returns
    -------
    np.ndarray
        The z-stack image as a numpy array.

    Raises
    -------
    ValueError
        If the image has less than 3 dimensions.
    """

    img = tifffile.imread(file_path)

    if len(img.shape) > 5:
        # determine in any of the dimensions is size of 1?
        img = np.squeeze(img)
    elif len(img.shape) < 3:
        raise ValueError(f"Image at {file_path} has less than 3 dimensions")

    if img.dtype != np.uint16:
        img = img.astype(np.uint16)

    return img
