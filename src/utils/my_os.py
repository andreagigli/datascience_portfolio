import os


def ensure_dir_exists(directory):
    """
    Ensures that a specified directory exists; if not, it creates the directory.

    This function checks if the directory specified by the `directory` parameter exists on the file system.
    If the directory does not exist, it is created along with any necessary intermediate directories. This function
    is particularly useful for setting up file storage paths in scripts and applications where the existence of
    the directory structure is a prerequisite.

    Args:
        directory (str): The path of the directory to check or create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

