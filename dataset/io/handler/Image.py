from typing import Any

from PIL import Image
from PIL.Image import Image as img

from dataset.io.Protocol import _StrOrBytesPath
from dataset.io.Register import ImageIOMeta

# get PIL_support_format using python3 -m PIL, all format in the list support open and save method
# tips: png and gif support save_all for multi-frame images, if you want to do that, set a custom image io for them
_PIL_support_format = [
    ".blp",
    ".bmp",
    ".bufr",
    ".dds",
    ".dib",
    ".eps",
    ".ps",
    ".gbr",
    ".gif",
    ".grib",
    ".h5",
    ".hdf",
    ".icns",
    ".ico",
    ".im",
    ".jfif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".j2c",
    ".j2k",
    ".jp2",
    ".jpc",
    ".jpf",
    ".jpx",
    ".msp",
    ".pcx",
    ".png",
    ".apng",
    ".pbm",
    ".pfm",
    ".pgm",
    ".pnm",
    ".ppm",
    ".qoi",
    ".bw",
    ".rgb",
    ".rgba",
    ".sgi",
    ".tga",
    ".icb",
    ".vda",
    ".vst",
    ".tif",
    ".tiff",
    ".webp",
    ".emf",
    ".wmf",
    ".xbm",
]


class BaseImage(metaclass=ImageIOMeta):  # type: ignore
    suffixes = _PIL_support_format

    @staticmethod
    def load(path: _StrOrBytesPath) -> Any:
        # There is try inside Image.open(), so in there, we don't use key word: try
        opened_image = Image.open(path)  # type: ignore
        return opened_image

    @staticmethod
    def write(path: _StrOrBytesPath, image: img) -> None:
        # same reason
        image.save(path)  # type: ignore
