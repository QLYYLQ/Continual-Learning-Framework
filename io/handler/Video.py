from typing import Any

from CLTrainingFramework.io.Register import create_io_registry
from CLTrainingFramework.io.Protocol import _StrOrBytesPath
from torchvision.io import VideoReader
VideoIOMeta = create_io_registry("Video")

class BaseVideo(metaclass=VideoIOMeta):
    suffixes = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv']
    _reader = VideoReader
    @staticmethod
    def load(path:_StrOrBytesPath,**kwargs):
        return VideoReader(path,**kwargs)
    def write(self, path:_StrOrBytesPath,obj:Any,**kwargs):
        raise NotImplementedError()