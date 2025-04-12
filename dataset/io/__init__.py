from dataset.io.IOProtocol import _MetaRegistry as ModalityRegistry
from dataset.io.IOProtocol import _SuffixRegistry as IORegistry

from dataset.io.Register import (
    IOProtocol,
    create_io_registry,
    MetaIO,
    ImageIOMeta,
    TextIOMeta,
)
from dataset.io.handler import BaseImage

__all__ = [
    "create_io_registry",
    "MetaIO",
    "IOProtocol",
    "IORegistry",
    "ModalityRegistry",
    "ImageIOMeta",
    "TextIOMeta",
    "BaseImage",
]
