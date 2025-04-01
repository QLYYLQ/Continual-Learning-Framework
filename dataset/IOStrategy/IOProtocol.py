from typing import List, Protocol, runtime_checkable, Union
from abc import ABC, abstractmethod
from os import PathLike


class LoadProtocol(Protocol):

    def load(self, file_name: Union[str, PathLike[str]]) -> List[List[int]]:
        ...


class WriteProtocol(Protocol):
    def write(self, file_name: Union[str, PathLike[str]]) -> None:
        ...


@runtime_checkable
class IOProtocol(LoadProtocol, WriteProtocol, Protocol):
    """
    Canonical IO strategy must have load and write methods.
    """
    pass


class IOMeta(type):
    """
    auto registry the IO class
    """
    _registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if issubclass(cls, IOProtocol):
            if hasattr(cls, "suffixes"):
                for suffix in cls.suffixes:
                    if suffix in IOMeta._registry:
                        existing = IOMeta._registry[suffix].__name__
                        raise ValueError(f".{suffix} is already registered by {existing}")
                    IOMeta._registry[suffix] = cls
        else:
            raise ValueError(f"Unsupported IO class {cls}")


class BaseIO(metaclass=IOMeta, ABC):
    suffixes: List[str] = []

    @abstractmethod
    def load(self, file_name: Union[str, PathLike[str]]) -> List[List[int]]:
        raise NotImplementedError

    @abstractmethod
    def write(self, file_name: Union[str, PathLike[str]]) -> None:
        raise NotImplementedError


