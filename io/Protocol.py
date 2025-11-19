from functools import wraps
from os import PathLike
from typing import (
    Protocol,
    runtime_checkable,
    Union,
    Any,
    Dict,
    Collection,
    Optional,
    TypedDict,
    Type,
    IO,
    ClassVar,
)
from weakref import WeakSet


class LoadProtocol(Protocol):
    def load(
        self, file_name: Union[str, PathLike[str]], modality: Optional[str] = None,**kwargs
    ) -> Any:
        ...


class WriteProtocol(Protocol):
    def write(self, file_name: Union[str, PathLike[str]], file: Any,**kwargs) -> Any:
        ...


@runtime_checkable
class IOProtocol(LoadProtocol, WriteProtocol, Protocol):
    """
    Canonical IO strategy must have load and write methods.
    """

    pass


class _T_ModalityRegistry(TypedDict):
    BaseIO: Type[IOProtocol]
    base_suffixes: Collection[str]
    Custom: Dict[str, Type[IOProtocol]]


def _suffix_usage(register_dict: dict) -> dict[str, set[str]]:
    """
    build a dict: {suffix, {modality_name}}
    """
    suffix_usage: dict[str, set[str]] = {}
    for modality_name, modality_registry in register_dict.items():
        if modality_registry:
            base_sfx = set(modality_registry["base_suffixes"])
            custom_sfx = set(modality_registry.get("Custom", {}).keys())
            all_sfx = base_sfx.union(custom_sfx)
            for suffix in all_sfx:
                suffix_usage.setdefault(suffix, set()).add(modality_name)
    return suffix_usage


def _update_register_dict(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self: Optional[RegisterDict] = None
        if args:
            self: "RegisterDict" = args[0]
            args = args[1:]
        else:
            self = kwargs.pop("self", None)
        assert self is not None
        out = func(self, *args, **kwargs)
        full_usage = _suffix_usage(self)
        collisions = {k: v for k, v in full_usage.items() if len(v) > 1}
        setattr(self, "collision_suffix", collisions)
        return out

    return wrapper


class RegisterDict(dict):
    def __init__(self, *args, **kwargs):
        super(RegisterDict, self).__init__(*args, **kwargs)
        self.collision_suffix: dict[str, set[str]] = {}
    def _collision_suffix_manually_update(self):
        """
        only be used when trigger can't work(for manual register)
        """
        full_usage = _suffix_usage(self)
        collisions = {k:v for k,v in full_usage.items() if len(v) >1}
        setattr(self, "collision_suffix", collisions)

    # callback function, tracing the suffix when updating the register
    # TODO 性能问题，每次都这样检查太傻了，有没有性能更高的解法？对于一个m模态，每个模态有n个后缀的字典，把一个O(1)的操作变成O(n*m)
    # TODO 考虑重写这些字典方法，按照修改的部分，只更新修改部分的映射关系就好
    __delitem__ = _update_register_dict(dict.__delitem__)
    __setitem__ = _update_register_dict(dict.__setitem__)
    update = _update_register_dict(dict.update)
    setdefault = _update_register_dict(dict.setdefault)
    pop = _update_register_dict(dict.pop)
    popitem = _update_register_dict(dict.popitem)
    clear = _update_register_dict(dict.clear)


_T_Registry = RegisterDict[str, _T_ModalityRegistry]


# TODO: using a more safety way to manage the registry, also add contextmanager to achieve IO sandbox mode


_SuffixRegistry: _T_Registry = RegisterDict()


@runtime_checkable
class _T_MetaIO(Protocol):
    _io_invalidation_counter: ClassVar[int]
    _io_cache: WeakSet[Type[IOProtocol]]
    _io_negative_cache: WeakSet[Type[IOProtocol]]
    _io_registry: WeakSet[Type[IOProtocol]]
    is_base: bool
    modality: str

    def _meta_register(
        self: Any,
        is_base: bool,
        registry: _T_ModalityRegistry,
        suffixes_list: Collection[str],
    ) -> None:
        ...

    @staticmethod
    def register(
        cls: Any, subclass: Any, suffixes: Optional[Collection[str]] = None
    ) -> Any:
        ...

    def __subclasscheck__(self: Any, subclass: Any) -> bool:
        ...

    def __instancecheck__(self: Any, instance: Any) -> bool:
        ...


@runtime_checkable
class _T_IOClass(
    _T_MetaIO,
    Protocol,
):
    _io_invalidation_cache_version: int

    def __subclasses__(self) -> Any:
        ...


_MetaRegistry: Dict[str, _T_MetaIO] = dict()


_StrOrBytesPath = Union[str, bytes, PathLike[str], IO[bytes]]

if __name__ == "__main__":
    register_dict = RegisterDict()
    register_dict["image"] = {
        "base_suffixes": ["png", "jpg", "jpeg"],
        "Custom": {"png": None},
    }
    register_dict["custom"] = {"base_suffixes": ["png"]}
    print(register_dict.collision_suffix)
