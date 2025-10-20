import importlib
import importlib.metadata
import os
import platform
from pathlib import Path
from typing import Optional
from version_helper import Version

PY_VERSION = Version(platform.python_version())

# import config

DILL_VERSION = Version(importlib.metadata.version("dill"))
FSSPEC_VERSION = Version(importlib.metadata.version("fsspec"))
PANDAS_VERSION = Version(importlib.metadata.version("pandas"))
PYARROW_VERSION = Version(importlib.metadata.version("pyarrow"))
TORCH_VERSION = Version(importlib.metadata.version("torch").split("+")[0])
if __name__ == "__main__":
    print(os.environ.get("USE_TORCH", "AUTO").upper())
    print(DILL_VERSION)
