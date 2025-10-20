from os import PathLike
from os.path import splitext
from pathlib import Path
from typing import Optional, Type, Any, TypeVar, Union, Literal


from CLTrainingFramework.io.Protocol import _T_ModalityRegistry, IOProtocol
from CLTrainingFramework.io.Protocol import _SuffixRegistry as IORegistry

_T_Return_Method = TypeVar("_T_Return_Method")

_T_Path = TypeVar("_T_Path", bound=Union[str, PathLike])

_mode_tye = Literal["tensor", "ndarray", "default"]


class IO:
    """
    Finding the IO class from _SuffixRegistry using the suffix of the file name.
    """

    def __init__(self,modality:Optional[str]=None) -> None:
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
        self._modality = modality
        self._previous_method: Optional[IOProtocol] = None
        self._previous_suffix: Optional[str] = None
        self._io_cache: dict[str, IOProtocol] = {}

    def load(
        self,
        path: _T_Path,
        modality: Optional[str] = None,
        collision_dict: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> Any:
        _collision_dict = {} if collision_dict is None else collision_dict
        if modality is None and self._modality is not None:
            io_method = self.get_io(path, self._modality, _collision_dict)
        else:
            io_method = self.get_io(path, modality, _collision_dict)
        return io_method.load(path,**kwargs)

    def write(
        self,
        path: _T_Path,
        data: Any,
        modality: Optional[str] = None,
        collision_dict: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> Any:
        _collision_dict = {} if collision_dict is None else collision_dict
        if modality is None and self._modality is not None:
            io_method = self.get_io(path, self._modality, _collision_dict)
        else:
            io_method = self.get_io(path, modality, _collision_dict)
        io_method.write(path, data,**kwargs)

    def get_io(
        self,
        path: _T_Path,
        modality: Optional[str] = None,
        collision_dict: Optional[dict[str, str]] = None,
    ) -> IOProtocol:
        suffix = self._get_path_suffix(path)
        suffix = suffix.lstrip(".")
        if suffix is self._previous_suffix:
            assert self._previous_method is not None
            return self._previous_method
        io_method, _ = self._get_io_method_by_suffix(suffix, modality, collision_dict)
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
        self,
        suffix: str,
        modality: Optional[str] = None,
        collision_dict: Optional[dict[str, str]] = None,
    ) -> tuple[IOProtocol, str]:
        """
        modality有两个作用，当suffix没有冲突的时候，modality作为一个快速索引的工具
        当suffix有冲突的时候，作为指示符
        在不使用modality的时候可能会出现一些意想不到的想象
        例如：你在两个不同的modality里注册了同样的后缀，表现为：a.c和b.c，在不指定modality的情况下返回是a.c还是b.c要看python的内部实现
            python 3.5及以前key的排序是跟着哈希值和一个随机种子走的，这导致每次启动程序的排序不一样，也就是返回的方法不一样
            python 3.6加入了保留插入顺序的技术，字典顺序是确定的，同时python的解释器是顺序执行代码的，所以其实我们注册表的顺序可以确定下来（虽然行为是确定的，但还是不推荐为一个后缀指定两种不同模态的读取方法以后传参不给模态名）
        说实话，你都设置让两种不同模态的方法读同一个后缀了，自己手动加modality不过分吧
        这就当特性吧 T_T
        Update:
            增加了冲突检测，如果后缀冲突，没有传模态就报错
        """
        # TODO 可以尝试增加一个方法检测，当注册表有不同模态读写同一后缀的不同方法后，load/write不传模态参数给他抛报错
        # check cache
        if modality:
            io_name = f"{modality}.{suffix}"
            if io_name in self._io_cache:
                return self._io_cache[io_name], suffix
        else:
            for cache_key in self._io_cache.keys():
                if cache_key.endswith(suffix):
                    return self._io_cache[cache_key], suffix
        # search in the registry
        io: Optional[Type[IOProtocol]] = None
        # check collision
        collision_suffix = IORegistry.collision_suffix
        if suffix in collision_suffix:
            if suffix not in collision_dict:
                raise ValueError(
                    f"File suffix: {suffix} in multiple modalities: {collision_suffix[suffix]}, you must set modality to write or load method"
                )
            else:
                io = self._get_io_method_with_modality(
                    suffix, IORegistry[collision_dict[suffix]]
                )
        if modality:
            io = self._get_io_method_with_modality(suffix, IORegistry[modality])
        else:
            for mod, modality_registry in IORegistry.items():
                if not io:
                    modality = mod
                    io = self._get_io_method_with_modality(suffix, modality_registry)
                else:
                    break
        if not io:
            raise ValueError(
                f"No IO method found for suffix: {suffix}, the register dict is: {IORegistry}"
            )
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
