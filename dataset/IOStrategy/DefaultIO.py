from os import PathLike
from typing import IO
from typing import Union

from PIL import Image
from PIL.Image import Image as img

from dataset.IOStrategy import ImageIOMeta

_StrOrBytesPath = Union[str, bytes, PathLike[str], PathLike[bytes], IO[bytes]]

# get PIL_support_format using python3 -m PIL, all format in the list support open and save method
# tips: png and gif support save_all for multi-frame images, if you want to do that, set a custom image
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

    def load(self, path: _StrOrBytesPath):
        opened_image = Image.open(path)  # type: ignore
        return opened_image

    def write(self, path: _StrOrBytesPath, image: img):
        image.save(path)  # type: ignore


if __name__ == "__main__":
    pass
