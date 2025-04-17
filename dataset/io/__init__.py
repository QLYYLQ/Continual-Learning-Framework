from dataset.io.Mapping import IO as io
from dataset.io.Protocol import IOProtocol
from dataset.io.Protocol import _MetaRegistry as ModalityRegistry
from dataset.io.Protocol import _SuffixRegistry as IORegistry
from dataset.io.Register import (
    create_io_registry,
    MetaIO,
    ImageIOMeta,
    TextIOMeta,
)
from dataset.io.handler import BaseText, YamlText, JsonText, BaseImage

IO = io()
__all__ = [
    "create_io_registry",
    "MetaIO",
    "IOProtocol",
    "IORegistry",
    "ModalityRegistry",
    "ImageIOMeta",
    "TextIOMeta",
    "BaseImage",
    "BaseText",
    "YamlText",
    "JsonText",
    "IO",
]
