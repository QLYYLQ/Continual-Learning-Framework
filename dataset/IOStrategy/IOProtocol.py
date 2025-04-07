from typing import List, Protocol, runtime_checkable, Union, Any, Dict, Iterable, Optional, TypedDict, TypeVar, Type
from os import PathLike
from weakref import WeakSet


class _T_ModalityRegistry(TypedDict):
    BaseIO: Type[Any]
    base_suffixes: List[str]
    Custom: Dict[str, Type[Any]]



_T_Registry = Dict[str, _T_ModalityRegistry]

_Registry: _T_Registry = dict()

class LoadProtocol(Protocol):

    def load(self, file_name: Union[str, PathLike[str]]) -> Any:
        ...


class WriteProtocol(Protocol):
    def write(self, file_name: Union[str, PathLike[str]]) -> Any:
        ...


@runtime_checkable
class IOProtocol(LoadProtocol, WriteProtocol, Protocol):
    """
    Canonical IO strategy must have load and write methods.
    """
    pass


class IOMeta(type):
    """
    Metaclass for BaseIO.
    Auto Register for subclass of BaseIO, manually register for subclass of IOProtocol
    This class is used inside Framework
    """

    _io_invalidation_counter = 0

    def __new__(mcls, name, bases, namespace):
        # check the basic attribute
        modality = getattr(mcls, "modality", None)
        is_base = getattr(mcls, "is_base", None)
        assert modality is not None and is_base is not None, "You should set modality and is_base"

        # create class
        cls = super().__new__(mcls, name, bases, namespace)

        # set attribute for class
        setattr(cls, "modality", modality)
        setattr(cls, "is_base", is_base)
        suffixes_list = getattr(cls, 'suffixes', [])
        suffixes_set = _check_suffixes(suffixes_list)
        if not suffixes_set and not is_base:
            raise ValueError(f"{name} must have suffixes attribute")
        # registering with different modality, first check exits modality
        if modality not in _Registry:
            _Registry[modality] = dict()
        mcls._meta_register(cls, is_base, _Registry[modality], suffixes_set)
        mcls.is_base = False
        # add base implementation
        cls._io_registry = WeakSet()
        # fast cache for subclass check
        cls._io_cache = WeakSet()
        cls._io_negative_cache = WeakSet()
        # version controller
        cls._io_invalidation_cache_version = IOMeta._io_invalidation_counter
        return cls

    def register(cls, subclass, suffixes: Optional[Iterable[str]] = None):
        """
        you can register a subclass with this method, without inheriting from BaseIO.
        You can also manually set suffixes
        example:
            from PIL import Image
            class MyIO(Image):
                def __init__(self, image):
                    self.suffixes = ['png', 'jpg', 'jpeg']
                def load():
                    ...
            # register a subclass with this method, only open .jpg suffix, not .png or other
            BaseIO.register(MyIO, suffixes=['.jpg'])
        """
        if not isinstance(subclass, type):
            raise TypeError("you can only register a class")
        if not issubclass(subclass, IOProtocol):
            raise TypeError("you can only register a subclass of IOProtocol(has load and write method)")
        if suffixes is None:
            suffixes = getattr(subclass, "suffixes", None)
        assert suffixes is not None, (f"if you want to register a subclass of {cls.__qualname__}, you must define "
                                      f"suffixes")
        suffixes = _check_suffixes(suffixes)
        # Already a subclass of cls, not need for registry
        if issubclass(subclass, cls):
            return subclass
        if issubclass(cls, subclass):
            raise RuntimeError(f"circular reference: {cls} is subclass of {subclass}")
        cls._io_registry.add(subclass)  # type: ignore
        # when you manually register class, that won't be base class
        cls._meta_register(subclass, False, _Registry[cls.modality], suffixes)  # type: ignore
        IOMeta._io_invalidation_counter += 1
        return subclass

    def __subclasscheck__(self, subclass):
        """
        rewrite issubclass(subclass, self)
        这里的检查机制是：先检查有没有在 self._io_cache的缓存中，再检查在不在self._io_negative_cache的缓存中（如果新加入了注册 ->
        两个常量值不匹配，更新self._io_negative_cache），随后检查 subclass hook和self.__mro__，都没查到就添加到
        self._io_negative_cache
        """
        if not isinstance(subclass, type):
            raise TypeError("arg 1 must be a class")
        if subclass in self._io_cache:
            return True
        if self._io_invalidation_cache_version < IOMeta._io_invalidation_counter:
            self._io_negative_cache = WeakSet()
            self._io_invalidation_cache_version = IOMeta._io_invalidation_counter
        elif subclass in self._io_negative_cache:
            return False
        # Check the subclass hook
        ok = self.__subclasshook__(subclass)
        if ok is not NotImplemented:
            assert isinstance(ok, bool)
            if ok:
                self._io_cache.add(subclass)
            else:
                self._io_negative_cache.add(subclass)
            return ok
        # check __mro__
        if self in getattr(subclass, "__mro__", ()):
            self._io_cache.add(subclass)
            return True
        for rcls in self._io_registry:
            if issubclass(subclass, rcls):
                self._io_cache.add(subclass)
                return True
        for scls in self.__subclasses__():
            if issubclass(subclass, scls):
                self._io_cache.add(subclass)
                return True
        self._io_negative_cache.add(subclass)
        return False

    @staticmethod
    def _meta_register(cls, is_base, registry: dict, suffixes_list: Iterable[str], ):
        """
        auto registry
        """
        if is_base:
            registry['base_suffixes'] = suffixes_list
            registry["BaseIO"] = cls
        else:
            for suffix in suffixes_list:
                if suffix not in registry.keys():
                    registry[suffix] = cls
                else:
                    # TODO: 这里需要添如果覆盖之前注册过的类以后的行为
                    registry[suffix] = cls


def create_io_registry(modality: str, is_base: bool = True, cls_name: Optional[str] = None) -> type:
    if cls_name is None:
        cls_name = f"{modality.capitalize()}IOMeta"
    attrs = {
        "modality": modality,
        "is_base": is_base,
        "__module__": __name__,
        "__qualname__": cls_name,
    }
    new_meta = type(cls_name, (IOMeta,), attrs)
    return new_meta


def _check_suffixes(suffixes: Iterable[str]) -> set:
    return_suffixes = []
    for suffix in suffixes:
        # The suffix may have uppercase letters, so don't make any adjustment here
        if suffix[0] == ".":
            # fix possible clerical errors by users
            return_suffixes.append(suffix.lstrip("."))
        else:
            return_suffixes.append(suffix)
    return set(return_suffixes)


if __name__ == "__main__":
    ImageIOMeta = create_io_registry("image", True)


    class BaseIO(metaclass=ImageIOMeta):  # type: ignore
        """
        You must add all possible suffixes to the list, and files with these suffixes should be able to be read and
        written by the methods you set
        """

        # should not use suffixes = [] with @dataclass() decorator, which create a shared list for all subclasses
        # function field() tell @dataclass() to create an attribute using list function for every subclass
        suffixes = ['png', 'jpg', 'jpeg']

        def check_path(self, path: Union[str, PathLike[str]]):
            pass

        def load(self, file_name: Union[str, PathLike[str]]) -> Any:
            raise NotImplementedError

        def write(self, file_name: Union[str, PathLike[str]]) -> Any:
            raise NotImplementedError


    class TryClass1(ImageIOMeta):  # type: ignore
        suffixes = ["try1"]

        def load(self, **kwargs):
            pass

        def write(self, **kwargs):
            pass


    class TryClass2:
        suffixes = ["try2"]

        def load(self, **kwargs):
            pass

        def write(self, **kwargs):
            pass


    class TryClass3:
        suffixes = ["try3"]

        def load(self, **kwargs):
            pass


    class TryClass4:
        def load(self, **kwargs):
            pass

        def write(self, **kwargs):
            pass


    # class BaseIO2(BaseIO):
    #     suffixes = ["io2"]
    #
    #     def load(self, **kwargs):
    #         pass
    #
    #     def write(self, **kwargs):
    #         pass
    #

    # print(dir(BaseIO))
    # BaseIO.register(TryClass1)
    BaseIO.register(TryClass2)  # type: ignore
    # print(issubclass(TryClass3, IOProtocol))
    BaseIO.register(TryClass4)  # type: ignore
    print(_Registry)
    # BaseIO.register(BaseIO2)
    # BaseIO.register(TryClass3)
