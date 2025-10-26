import copy
import itertools
from dataclasses import is_dataclass, fields


def _deepcopy(x, memo: dict):
    """deepcopy a regular class instance"""
    cls = x.__class__
    # avoid __init__ method
    result = cls.__new__(cls)
    memo[id(x)] = result
    for k, v in x.__dict__.items():
        setattr(result, k, copy.deepcopy(v, memo))
    return result


def _check_datclass_instance(obj):
    return is_dataclass(obj) and not isinstance(obj, type)


def _inner_as_dict(obj):
    if _check_datclass_instance(obj):
        result = {}
        for f in fields(obj):
            value = _inner_as_dict(getattr(obj, f.name))
            if (
                not f.init
                or value != f.default
                or f.metadata.get("include_in_as_dict_even_if_is_default", False)
            ):
                result[f.name] = value
        return result
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # obj is a namedtuple
        return type(obj)(*[_inner_as_dict(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_inner_as_dict(v) for v in obj)
    elif isinstance(obj, dict):
        return {_inner_as_dict(k): _inner_as_dict(v) for k, v in obj.items()}
    else:
        return copy.deepcopy(obj)


def as_dict(obj)->dict:
    """
    Convert an object to its dictionary representation
    base on https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
    """
    if not isinstance(obj, dict) and not _check_datclass_instance(obj):
        raise TypeError("%s is not a dataclass instance or dict" % obj)
    return _inner_as_dict(obj)


def unique_values(values):
    seen = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            yield v


def zip_dict(*dicts):
    for k in unique_values(itertools.chain(*dicts)):
        yield k, tuple(d[k] for d in dicts)


if __name__ == "__main__":
    schema = {"id": "int64", "audio": "Audio", "image": "Image"}
    obj = {"id": [12, 23], "audio": "path/to/file.wav", "image": "path/to/image.jpg"}
    iterator = zip_dict(schema, obj)
    print(list(iterator))


def no_op_if_value_is_null(func):
    """If the value is None, return None, else call `func`."""

    def wrapper(value):
        return func(value) if value is not None else None

    return wrapper

def first_non_null_non_empty_value(iterable):
    """Return the index and the value of the first non-null non-empty value in the iterable. If all values are None or empty, return -1 as index."""
    for i, value in enumerate(iterable):
        if value is not None and not (isinstance(value, (dict, list)) and len(value) == 0):
            return i, value
    return -1, None


class NonMutableDict(dict):
    """Dict where keys can only be added but not modified.

    Will raise an error if the user try to overwrite one key. The error message
    can be customized during construction. It will be formatted using {key} for
    the overwritten key.
    """

    def __init__(self, *args, **kwargs):
        self._error_msg = kwargs.pop(
            "error_msg",
            "Try to overwrite existing key: {key}",
        )
        if kwargs:
            raise ValueError("NonMutableDict cannot be initialized with kwargs.")
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(self._error_msg.format(key=key))
        return super().__setitem__(key, value)

    def update(self, other, **kwargs):
        if any(k in self for k in other):
            raise ValueError(self._error_msg.format(key=set(self) & set(other)))
        return super().update(other,**kwargs)
