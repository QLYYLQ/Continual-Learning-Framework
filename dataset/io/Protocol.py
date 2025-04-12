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
    Callable,
    IO,
)



class _T_ModalityRegistry(TypedDict):
    BaseIO: Type[Any]
    base_suffixes: Collection[str]
    Custom: Dict[str, Type[Any]]


_T_Registry = Dict[str, _T_ModalityRegistry]
# TODO: using a more safety way to manage the registry, also add contextmanager to achieve IO sandbox mode
_SuffixRegistry: _T_Registry = dict()


class _T_MetaIO(Protocol):
    _io_invalidation_counter: int
    _io_cache: Dict
    _io_negative_cache: Dict
    _io_registry: Dict
    _meta_register: Callable[
        [Type[Any], bool, _T_ModalityRegistry, Collection[str]], None
    ]
    is_base: bool
    modality: str
    register: Callable[[Type[Any], Optional[Collection[str]]], Type[Any]]
    __subclasshook__: Callable[[Any], bool]  # type: ignore
    __instancecheck__: Callable[[Any], bool]

    @property
    def io_invalidation_counter(self):
        return self._io_invalidation_counter


_MetaRegistry: Dict[str, Union[type, _T_MetaIO]] = dict()


class LoadProtocol(Protocol):
    def load(self, file_name: Union[str, PathLike[str]]) -> Any:
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


_StrOrBytesPath = Union[str, bytes, PathLike[str], PathLike[bytes], IO[bytes]]
