from os import PathLike
from os.path import splitext
from pathlib import Path
from typing import Optional, Type, Any, TypeVar, Union, Literal


from dataset.io.Protocol import _T_ModalityRegistry, IOProtocol, _SuffixRegistry

_T_Return_Method = TypeVar("_T_Return_Method")

_T_Path = TypeVar("_T_Path", bound=Union[str, PathLike])

_mode_tye = Literal["tensor", "ndarray", "default"]


class IO:
    """
    Finding the IO class from _SuffixRegistry using the suffix of the file name.
    """

    def __init__(self) -> None:
        """
        Represents the initialization of an instance of a class. Sets up an internal dictionary
        to cache IOProtocol mappings identified by string keys: {}. This is used to store and
        retrieve input/output operations efficiently.

        Attributes:
            self._io_cache (dict[str, Type[IOProtocol]]): A dictionary used for caching IOProtocol
            instances with string keys.
            self._previous_method (IOProtocol): Previous IOProtocol instance
            self._previous_suffix (str): Previous path suffix
        """
        self._previous_method: Optional[IOProtocol] = None
        self._previous_suffix: Optional[str] = None
        self._io_cache: dict[str, IOProtocol] = {}

    def load(self, path: _T_Path, modality: Optional[str] = None) -> Any:
        io_method = self.get_io(path, modality)
        return io_method.load(path)

    def write(self, path: _T_Path, data: Any, modality: Optional[str] = None) -> Any:
        io_method = self.get_io(path, modality)
        io_method.write(path, data)

    def get_io(self, path: _T_Path, modality: Optional[str] = None) -> IOProtocol:
        suffix = self._get_path_suffix(path)
        suffix = suffix.lstrip(".")
        if suffix is self._previous_suffix:
            assert self._previous_method is not None
            return self._previous_method
        io_method, _ = self._get_io_method_by_suffix(path, suffix, modality)
        return io_method

    @staticmethod
    def _get_path_suffix(path: _T_Path) -> str:
        if isinstance(path, Path):
            return path.suffix
        elif isinstance(path, str):
            _, suffix = splitext(path)
            return suffix
        else:
            try:
                path_str = str(path)
                _, suffix = splitext(path_str)
                return suffix
            except TypeError:
                raise TypeError(f"Unsupported path type: {type(path)}")

    def _get_io_method_by_suffix(
        self, path: _T_Path, suffix: str, modality: Optional[str] = None
    ) -> tuple[IOProtocol, str]:
        # check cache
        if modality:
            io_name = f"{modality}.{suffix}"
        else:
            io_name = None
        if io_name in self._io_cache:
            return self._io_cache[io_name], suffix
        if not io_name:
            for cache_key in self._io_cache.keys():
                if cache_key.endswith(suffix):
                    return self._io_cache[cache_key], suffix
        # search in the registry
        io: Optional[Type[IOProtocol]] = None
        if modality:
            io = self._get_io_method_with_modality(suffix, _SuffixRegistry[modality])
        else:
            for mod, modality_registry in _SuffixRegistry.items():
                if not io:
                    modality = mod
                    io = self._get_io_method_with_modality(suffix, modality_registry)
                else:
                    break
        if not io:
            raise ValueError(f"No IO method found for suffix: {suffix}")
        _io = io()
        self._io_cache[f"{modality}.{suffix}"] = _io
        return _io, suffix

    @staticmethod
    def _get_io_method_with_modality(
        suffix: str, modality_registry: _T_ModalityRegistry
    ) -> Optional[Type[IOProtocol]]:
        io_class: Optional[Type[IOProtocol]] = None
        if modality_registry.get("Custom"):
            custom_registry = modality_registry.get("Custom")
            if suffix in custom_registry:  # type: ignore
                io_class = custom_registry[suffix]  # type: ignore
        if not io_class:
            if suffix in modality_registry["base_suffixes"]:
                io_class = modality_registry["BaseIO"]
        return io_class

    def delete_cache(self, name: str) -> None:
        try:
            del self._io_cache[name]
        except KeyError:
            print(f"No cache found for {name}")
