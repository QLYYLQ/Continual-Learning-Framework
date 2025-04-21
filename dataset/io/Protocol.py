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
        self, file_name: Union[str, PathLike[str]], modality: Optional[str] = None
    ) -> Any:
        ...


class WriteProtocol(Protocol):
    def write(self, file_name: Union[str, PathLike[str]], file: Any) -> Any:
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


_T_Registry = Dict[str, _T_ModalityRegistry]
# TODO: using a more safety way to manage the registry, also add contextmanager to achieve IO sandbox mode
_SuffixRegistry: _T_Registry = dict()


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


_StrOrBytesPath = Union[str, bytes, PathLike[str], PathLike[bytes], IO[bytes]]
