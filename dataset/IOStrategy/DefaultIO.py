import numpy as np
from PIL import Image
from typing import Union
from os import PathLike
from dataset.IOStrategy.IOProtocol import BaseIO


class ImageIO(BaseIO):
    suffixes = ['png', 'jpg', 'jpeg']

    def load(self, path: Union[str, PathLike]) -> np.ndarray:
        image = Image.open(path)
        return np.array(image)

    def write(self, file_name: Union[str, PathLike[str]]) -> np.ndarray:
        return np.array(Image.open(file_name))

if __name__ == '__main__':
    from dataset.IOStrategy import IORegister
    print(IORegister)
    print("yes")
    import os
    import sys
    # for i in os.environ["PATH"].split(":"):
    #     print(i)
    for i in sys.path:
        print(sys.path)