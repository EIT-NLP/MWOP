"""Resolve optional local copies of the public benchmark datasets."""
import os
from pathlib import Path


def resolve_dataset_path(path):
    if not path:
        return path
    expanded = os.path.expanduser(os.path.expandvars(path))
    if Path(expanded).exists():
        return expanded
    root = os.environ.get("LMMS_DATA_ROOT")
    if root:
        local = Path(os.path.expanduser(os.path.expandvars(root))) / Path(path).name
        if local.is_dir():
            return str(local.resolve())
    return path
