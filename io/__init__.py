from CLTrainingFramework.io.Mapping import IO as io_builder
from CLTrainingFramework.io.Protocol import IOProtocol
from CLTrainingFramework.io.Protocol import _MetaRegistry as ModalityRegistry
from CLTrainingFramework.io.Protocol import _SuffixRegistry as IORegistry
from CLTrainingFramework.io.Protocol import RegisterDict
from CLTrainingFramework.io.Register import (
    create_io_registry,
    MetaIO,
)
from .handler import BaseText, YamlText, JsonText, BaseImage, ImageIOMeta,TextIOMeta

IO = io_builder()
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
    "io_builder",
    "RegisterDict",
]
