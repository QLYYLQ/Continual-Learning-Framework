from typing import List, Protocol, runtime_checkable, Union
from abc import ABC, abstractmethod
from os import PathLike
import torch


class LoadProtocol(Protocol):

    def load(self, file_name: Union[str, PathLike[str]]) -> List[List[int]]:
        ...

if __name__ == '__main__':
    import inspect
    # print(inspect.get_annotations(LoadProtocol.load))
    # from torch import nn
    # self = object()
    # head_channels = 256
    # use_bias = True
    # classes = [16,1,1,1]
    # self.cls = nn.ModuleList([nn.Conv2d(head_channels, c, 1, bias=use_bias) for c in classes])
    print(torch.cuda.nccl.version())

