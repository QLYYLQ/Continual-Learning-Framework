import os
from typing import Protocol, runtime_checkable, Union, Any
from os import PathLike


class _ReplayStrategy(Protocol):
    """
    define the interface for all replay strategies, which need use fetch_data() to return replay data(can be multi-thread)
    Must be inherited from Protocol
    """

    def fetch_data(self) -> Any:
        pass


@runtime_checkable
class _DataReplay(Protocol):
    def download_buffer(self, download_path: Union[PathLike, str]):
        """
        Downloads the data buffer to the given path.
        tips:
            1. if you are using python 3.10 or higher, download_path: str | PathLike is a more pythonic way for typing.
            2. maybe I mention in README, you shouldn't use python 3.8 or lower version with the reason that there are
            some genericity in the framework. Maybe I will fix it later
        """
        pass

    def update_buffer(self, strategy: _ReplayStrategy, sign):
        """
        Updates the data buffer with the given strategy and sign
        the strategy may be a multi-thread process, so buffer is necessary for strategy.
        """
        pass


@runtime_checkable
class OnlineReplayStrategy(_ReplayStrategy, Protocol):
    base_url: str = None
    api_key: str = None

    def request_with_message(self, message):
        pass


@runtime_checkable
class LocalReplayStrategy(_ReplayStrategy, Protocol):
    pass


@runtime_checkable
class GenericReplayStrategy(_ReplayStrategy, Protocol):
    def generic_with_gan(self, gan_model) -> Any:
        pass
