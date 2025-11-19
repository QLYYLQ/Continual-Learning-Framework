# 尝试导入 Cython 优化版本，如果失败则回退到纯 Python 版本
try:
    print("success import cython\n\n\n")
    from CLTrainingFramework.io.Mapping_cython import IO as io_builder
except ImportError:
    print("NO!!!\n\n\n\n\nNO!!!")
    # Cython 版本不可用，使用纯 Python 版本
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
