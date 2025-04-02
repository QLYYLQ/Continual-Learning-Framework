from typing import List, Protocol, runtime_checkable, Union, Any
from abc import ABC, abstractmethod
from os import PathLike



class LoadProtocol(Protocol):

    def load(self, file_name: Union[str, PathLike[str]]) -> Any:
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


def registry():
    return IOMeta._registry


class IOMeta(type):
    """
    auto registry the IO class
    """
    _registry = {}
    @property
    def registry(cls):
        return IOMeta._registry
    @registry.setter
    def registry(cls, value):
        raise TypeError("you can't change the registry dict of the IO class")

    @registry.deleter
    def registry(cls):
        raise TypeError("you can't change the registry dict of the IO class")

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



class BaseIO(metaclass=IOMeta):
    """
    You must add all possible suffixes to the list, and files with these suffixes should be able to be read and written
    by the methods you set
    """
    # should not use suffixes = [] with @dataclass() decorator, which create a shared list for all subclasses
    # function field() tell @dataclass() to create an attribute using list function for every subclass

    suffixes: List[str] = ["test"]

    @abstractmethod
    def load(self, file_name: Union[str, PathLike[str]]) -> List[List[int]]:
        raise NotImplementedError

    @abstractmethod
    def write(self, file_name: Union[str, PathLike[str]]) -> None:
        raise NotImplementedError

if __name__ == "__main__":
    print(issubclass(BaseIO, IOProtocol))