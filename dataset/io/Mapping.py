from os import PathLike
from typing import Optional, Type, Any

from dataset.io.Protocol import _T_ModalityRegistry, IOProtocol, _SuffixRegistry


class IO:
    """
    Finding the IO class from _SuffixRegistry using the suffix of the file name.
    """

    def __init__(self):
        """
        Represents the initialization of an instance of a class. Sets up an internal dictionary
        to cache IOProtocol mappings identified by string keys: {modality.suffix: instance}. This is used to store and
        retrieve input/output operations efficiently.

        Attributes:
            self._io_cache (dict[str, Type[IOProtocol]]): A dictionary used for caching IOProtocol
            instances with string keys.
        """
        self._io_cache: dict[str, IOProtocol] = {}  # type: ignore

    def load(self, path: PathLike[str], modality: Optional[str] = None) -> Any:
        io_method = self.get_io(path)
        return io_method.load(path)

    def write(
        self, path: PathLike[str], data: Any, modality: Optional[str] = None
    ) -> Any:
        io_method = self.get_io(path, modality)
        io_method.write(path, data)

    def get_io(self, path: PathLike[str], modality: Optional[str] = None) -> IOProtocol:
        io_method = self._get_io_method_by_suffix(path, modality)
        return io_method

    def _get_io_method_by_suffix(
        self, path: PathLike[str], modality: Optional[str] = None
    ) -> IOProtocol:
        suffix = path.split(".")[-1]  # type: ignore
        # check cache
        if modality:
            io_name = f"{modality}.{suffix}"
        else:
            io_name = None
        if io_name in self._io_cache:
            return self._io_cache[io_name]
        if not io_name:
            for cache_key in self._io_cache.keys():
                if cache_key.endswith(suffix):
                    return self._io_cache[cache_key]
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
        return _io

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

    def delete_cache(self, name: str):
        try:
            del self._io_cache[name]
        except KeyError:
            print(f"No cache found for {name}")
