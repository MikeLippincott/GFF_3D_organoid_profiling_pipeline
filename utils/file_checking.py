import pathlib
from typing import Tuple


def check_number_of_files(
    directory: pathlib.Path, n_files: int
) -> Tuple[str, int] | None:
    """
    Check if the number of files in a directory is equal to a given number.

    Parameters
    ----------
    directory : pathlib.Path
        Specified directory to check file number.
    n_files : int
        The expected number of files in the directory.

    Returns
    -------
    Tuple[str, int] | None
        If the number of files does not match `n_files`, returns a tuple with the directory
        name and the actual number of files found. If the number matches, returns None.
    """
    files = list(directory.glob("*"))
    files = [f for f in files if f.is_file()]
    if len(files) != n_files:
        print(
            f"{directory.name} expected {n_files} files, but found {len(files)} files."
        )
        return directory.name, len(files)
    return None
