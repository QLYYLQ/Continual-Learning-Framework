import copy
import itertools
import multiprocessing.pool
import warnings
import queue
from dataclasses import is_dataclass, fields
from typing import Union, Callable, Any, Optional, TypeVar, Iterable

import multiprocess.pool
import numpy as np
from tqdm.asyncio import tqdm

from CLTrainingFramework.dataset.arrow_handler.parallel import parallel_map
from CLTrainingFramework.utils import logging
from reference.datasets import tqdm as hf_tqdm


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


def as_dict(obj) -> dict:
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


def first_non_null_value(iterable):
    """Return the index and the value of the first non-null value in the iterable. If all values are None, return -1 as index."""
    for i, value in enumerate(iterable):
        if value is not None:
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
        return super().update(other, **kwargs)


def convert_file_size_to_int(size: Union[int, str]) -> int:
    """
    Converts a size expressed as a string with digits a unit (like `"50MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("PIB"):
        return int(size[:-3]) * (2 ** 50)
    if size.upper().endswith("TIB"):
        return int(size[:-3]) * (2 ** 40)
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2 ** 30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2 ** 20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2 ** 10)
    if size.upper().endswith("PB"):
        int_size = int(size[:-2]) * (10 ** 15)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("TB"):
        int_size = int(size[:-2]) * (10 ** 12)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10 ** 9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10 ** 6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10 ** 3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError(f"`size={size}` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


def _single_map_nested(args):
    """Apply a function recursively to each element of a nested data struct."""
    function, data_struct, batched, batch_size, types, rank, disable_tqdm, desc = args

    # Singleton first to spare some computation
    if not isinstance(data_struct, dict) and not isinstance(data_struct, types):
        if batched:
            return function([data_struct])[0]
        else:
            return function(data_struct)
    if (
            batched
            and not isinstance(data_struct, dict)
            and isinstance(data_struct, types)
            and all(not isinstance(v, (dict, types)) for v in data_struct)
    ):
        return [mapped_item for batch in iter_batched(data_struct, batch_size) for mapped_item in function(batch)]

    # Reduce logging to keep things readable in multiprocessing with tqdm
    if rank is not None and logging.get_verbosity() < logging.WARNING:
        logging.set_verbosity_warning()
    # Print at least one thing to fix tqdm in notebooks in multiprocessing
    # see https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308
    if rank is not None and not disable_tqdm and any("notebook" in tqdm_cls.__name__ for tqdm_cls in tqdm.__mro__):
        print(" ", end="", flush=True)

    # Loop over single examples or batches and write to buffer/file if examples are to be updated
    pbar_iterable = data_struct.items() if isinstance(data_struct, dict) else data_struct
    pbar_desc = (desc + " " if desc is not None else "") + "#" + str(rank) if rank is not None else desc
    with hf_tqdm(pbar_iterable, disable=disable_tqdm, position=rank, unit="obj", desc=pbar_desc) as pbar:
        if isinstance(data_struct, dict):
            return {
                k: _single_map_nested((function, v, batched, batch_size, types, None, True, None)) for k, v in pbar
            }
        else:
            mapped = [_single_map_nested((function, v, batched, batch_size, types, None, True, None)) for v in pbar]
            if isinstance(data_struct, list):
                return mapped
            elif isinstance(data_struct, tuple):
                return tuple(mapped)
            else:
                return np.array(mapped)


def map_nested(
        function: Callable[[Any], Any],
        data_struct: Any,
        dict_only: bool = False,
        map_list: bool = True,
        map_tuple: bool = False,
        map_numpy: bool = False,
        num_proc: Optional[int] = None,
        parallel_min_length: int = 2,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        types: Optional[tuple] = None,
        disable_tqdm: bool = True,
        desc: Optional[str] = None,
) -> Any:
    """Apply a function recursively to each element of a nested data struct.

    Use multiprocessing if num_proc > 1 and the length of data_struct is greater than or equal to
    `parallel_min_length`.

    <Changed version="2.5.0">

    Before version 2.5.0, multiprocessing was not used if `num_proc` was greater than or equal to ``len(iterable)``.

    Now, if `num_proc` is greater than or equal to ``len(iterable)``, `num_proc` is set to ``len(iterable)`` and
    multiprocessing is used.

    </Changed>

    Args:
        function (`Callable`): Function to be applied to `data_struct`.
        data_struct (`Any`): Data structure to apply `function` to.
        dict_only (`bool`, default `False`): Whether only apply `function` recursively to `dict` values in
            `data_struct`.
        map_list (`bool`, default `True`): Whether also apply `function` recursively to `list` elements (besides `dict`
            values).
        map_tuple (`bool`, default `False`): Whether also apply `function` recursively to `tuple` elements (besides
            `dict` values).
        map_numpy (`bool, default `False`): Whether also apply `function` recursively to `numpy.array` elements (besides
            `dict` values).
        num_proc (`int`, *optional*): Number of processes.
            The level in the data struct used for multiprocessing is the first level that has smaller sub-structs,
            starting from the root.
        parallel_min_length (`int`, default `2`): Minimum length of `data_struct` required for parallel
            processing.
            <Added version="2.5.0"/>
        batched (`bool`, defaults to `False`):
            Provide batch of items to `function`.
            <Added version="2.19.0"/>
        batch_size (`int`, *optional*, defaults to `1000`):
            Number of items per batch provided to `function` if `batched=True`.
            If `batch_size <= 0` or `batch_size == None`, provide the full iterable as a single batch to `function`.
            <Added version="2.19.0"/>
        types (`tuple`, *optional*): Additional types (besides `dict` values) to apply `function` recursively to their
            elements.
        disable_tqdm (`bool`, default `True`): Whether to disable the tqdm progressbar.
        desc (`str`, *optional*): Prefix for the tqdm progressbar.

    Returns:
        `Any`
    """
    if types is None:
        types = []
        if not dict_only:
            if map_list:
                types.append(list)
            if map_tuple:
                types.append(tuple)
            if map_numpy:
                types.append(np.ndarray)
        types = tuple(types)

    # Singleton
    if not isinstance(data_struct, dict) and not isinstance(data_struct, types):
        if batched:
            data_struct = [data_struct]
        mapped = function(data_struct)
        if batched:
            mapped = mapped[0]
        return mapped

    iterable = list(data_struct.values()) if isinstance(data_struct, dict) else data_struct

    if num_proc is None:
        num_proc = 1
    if any(isinstance(v, types) and len(v) > len(iterable) for v in iterable):
        mapped = [
            map_nested(
                function=function,
                data_struct=obj,
                num_proc=num_proc,
                parallel_min_length=parallel_min_length,
                batched=batched,
                batch_size=batch_size,
                types=types,
            )
            for obj in iterable
        ]
    elif num_proc != -1 and num_proc <= 1 or len(iterable) < parallel_min_length:
        if batched:
            if batch_size is None or batch_size <= 0:
                batch_size = max(len(iterable) // num_proc + int(len(iterable) % num_proc > 0), 1)
            iterable = list(iter_batched(iterable, batch_size))
        mapped = [
            _single_map_nested((function, obj, batched, batch_size, types, None, True, None))
            for obj in hf_tqdm(iterable, disable=disable_tqdm, desc=desc)
        ]
        if batched:
            mapped = [mapped_item for mapped_batch in mapped for mapped_item in mapped_batch]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".* is experimental and might be subject to breaking changes in the future\\.$",
                category=UserWarning,
            )
            if batched:
                if batch_size is None or batch_size <= 0:
                    batch_size = len(iterable) // num_proc + int(len(iterable) % num_proc > 0)
                iterable = list(iter_batched(iterable, batch_size))
            mapped = parallel_map(
                function, iterable, num_proc, batched, batch_size, types, disable_tqdm, desc, _single_map_nested
            )
            if batched:
                mapped = [mapped_item for mapped_batch in mapped for mapped_item in mapped_batch]

    if isinstance(data_struct, dict):
        return dict(zip(data_struct.keys(), mapped))
    else:
        if isinstance(data_struct, list):
            return mapped
        elif isinstance(data_struct, tuple):
            return tuple(mapped)
        else:
            return np.array(mapped)


T = TypeVar("T")


def iter_batched(iterable: Iterable[T], n: int) -> Iterable[list[T]]:
    if n < 1:
        raise ValueError(f"Invalid batch size {n}")
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


# ------- parallel helper

def _get_pid(pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool], ) -> set[int]:
    return {i.pid for i in pool._pool}


def _generator_queue_helper(queue: queue.Queue, func:Callable[...,Iterable[T]],kwargs:dict):
    for i, result in enumerate(func(**kwargs)):
        queue.put(result)


def parallel_flatmap_unordered(pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool],
                               func: Callable[..., Iterable[T]], *, iterable_kwargs: Iterable[dict]) -> Iterable[T]:
    init_pid = _get_pid(pool)
    _is_pool_pid_changed = False
    manage_cls = multiprocessing.Manager if isinstance(pool, multiprocessing.pool.Pool) else multiprocess.Manger
    with manage_cls() as manager:
        queue = manager.Queue()
        async_results = [
            pool.apply_async(_generator_queue_helper,(queue,func,i) )
            for i in iterable_kwargs
        ]
        try:
            while True:
                try:
                    yield queue.get(timeout = 0.05)
                except queue.Empty:
                    if all(i.ready() for i in async_results) and queue.empty():
                        break
                if _get_pid(pool)!=init_pid:
                    _is_pool_pid_changed = True
                    raise RuntimeError(f"有subprocesses在map的时候死了，你可能需要停止使用multiprocessing")

        finally:
            if not _is_pool_pid_changed:
                # we get the result in case there's an error to raise
                [async_result.get(timeout=0.05) for async_result in async_results]
