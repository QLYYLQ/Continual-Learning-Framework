from typing import Optional, Collection

from dataset.io.Protocol import (
    _SuffixRegistry,
    _MetaRegistry,
    IOProtocol,
    _T_ModalityRegistry,
)


class MetaIO(type):
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
        assert (
            modality is not None and is_base is not None
        ), "You should set modality and is_base"

        # create class
        cls = super().__new__(mcls, name, bases, namespace)

        # set attribute for class
        setattr(cls, "modality", modality)
        setattr(cls, "is_base", is_base)
        suffixes_list = getattr(cls, "suffixes", [])
        suffixes_set = _check_suffixes(suffixes_list)
        if not suffixes_set and not is_base:
            raise ValueError(f"{name} must have suffixes attribute")
        # registering with different modality, first check exits modality
        if modality not in _SuffixRegistry:
            _SuffixRegistry[modality] = dict()
        mcls._meta_register(cls, is_base, _SuffixRegistry[modality], suffixes_set)
        mcls.is_base = False
        # add base implementation
        cls._io_registry = dict()
        # fast cache for subclass check
        cls._io_cache = dict()
        cls._io_negative_cache = dict()
        # version controller
        cls._io_invalidation_cache_version = _MetaRegistry[
            modality
        ]._io_invalidation_counter
        return cls

    def register(cls, subclass, suffixes: Optional[Collection[str]] = None):
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
            raise TypeError(
                "you can only register a subclass of IOProtocol(has load and write method)"
            )
        if suffixes is None:
            suffixes = getattr(subclass, "suffixes", None)
        assert suffixes is not None, (
            f"if you want to register a subclass of {cls.__qualname__}, you must define "
            f"suffixes"
        )
        suffixes = _check_suffixes(suffixes)
        # Already a subclass of cls, not need for registry
        if issubclass(subclass, cls):
            return subclass
        if issubclass(cls, subclass):
            raise RuntimeError(f"circular reference: {cls} is subclass of {subclass}")
        cls._io_registry.add(subclass)  # type: ignore
        # when you manually register class, that won't be base class
        cls._meta_register(subclass, False, _SuffixRegistry[cls.modality], suffixes)  # type: ignore
        _MetaRegistry[cls.modality]._io_invalidation_counter += 1  # type: ignore
        return subclass

    def __subclasscheck__(self, subclass):
        """
        rewrite issubclass(subclass, self)
        这里的检查机制是：先检查有没有在 self._io_cache的缓存中，再检查在不在self._io_negative_cache的缓存中（如果新加入了注册 ->
        两个常量值不匹配，更新self._io_negative_cache），随后检查 subclass hook和self.__mro__，都没查到就添加到
        self._io_negative_cache
        """
        _Meta = _MetaRegistry[self.modality]
        if not isinstance(subclass, type):
            raise TypeError("arg 1 must be a class")
        if subclass in self._io_cache:
            return True
        if self._io_invalidation_cache_version < _Meta._io_invalidation_counter:
            self._io_negative_cache = dict()
            self._io_invalidation_cache_version = _Meta._io_invalidation_counter
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

    def __instancecheck__(self, instance):
        """
        rewrite isinstance(instance, self)
        """
        _Meta = _MetaRegistry[self.modality]
        subclass = instance.__class__
        # using cache for quick check
        if subclass in self._io_cache:
            return True
        subtype = type(instance)
        if subtype in subclass:
            # when negative_cache is the newest version
            if (
                self._io_invalidation_cache_version == _Meta._io_invalidation_counter
                and subtype in self._io_negative_cache
            ):
                return False
            return self.__subclasscheck__(subtype)
        return any(self.__subclasscheck__(c) for c in (subclass, subtype))

    @staticmethod
    def _meta_register(
        cls,
        is_base,
        registry: _T_ModalityRegistry,
        suffixes_list: Collection[str],
    ):
        """
        auto registry
        """
        if is_base:
            registry["base_suffixes"] = suffixes_list
            registry["BaseIO"] = cls
            registry["Custom"] = dict()
        else:
            for suffix in suffixes_list:
                if suffix not in registry.keys():
                    registry["Custom"][suffix] = cls
                else:
                    # TODO: 这里需要添如果覆盖之前注册过的类以后的行为
                    registry["Custom"][suffix] = cls


def create_io_registry(
    modality: str, is_base: bool = True, cls_name: Optional[str] = None
) -> type:
    if cls_name is None:
        cls_name = f"{modality.capitalize()}MetaIO"
    attrs = {
        "modality": modality,
        "is_base": is_base,
        "__module__": __name__,
        "__qualname__": cls_name,
    }
    new_meta = type(cls_name, (MetaIO,), attrs)
    _MetaRegistry[modality] = new_meta
    return new_meta


def _check_suffixes(suffixes: Collection[str]) -> set:
    return_suffixes = []
    for suffix in suffixes:
        # The suffix may have uppercase letters, so don't make any adjustment here
        if suffix[0] == ".":
            # fix possible clerical errors by users
            return_suffixes.append(suffix.lstrip("."))
        else:
            return_suffixes.append(suffix)
    return set(return_suffixes)


ImageIOMeta = create_io_registry("Image")
TextIOMeta = create_io_registry("Text")
