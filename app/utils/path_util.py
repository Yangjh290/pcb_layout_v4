import os


def convert_relative_path(path: str):
    """
    Convert a relative path to an absolute path.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, path)