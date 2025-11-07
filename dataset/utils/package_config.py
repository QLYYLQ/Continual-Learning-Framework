# Mine
import importlib
import importlib.metadata
import importlib.util
from packaging import version
PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    try:
        TORCH_VERSION = version.parse(importlib.metadata.version("torch"))
    except importlib.metadata.PackageNotFoundError:
        pass
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None
