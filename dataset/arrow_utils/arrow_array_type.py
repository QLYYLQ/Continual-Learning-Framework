# modified from huggingface

import json
from typing import Optional, Literal, Union, Any
import numpy as np
import pyarrow as pa
import pandas as pd
from dataclasses import field, dataclass
from .arrow_helper import str_to_arrow_type, _is_zero_copy_only
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype
from collections.abc import Sequence as SequenceCollections


class ArrayExtensionArray(pa.ExtensionArray):
    def __array__(self):
        zero_copy_only = _is_zero_copy_only(self.storage.type, unnest=True)
        return self.to_numpy(zero_copy_only=zero_copy_only)

    def __getitem__(self, i):
        return self.storage[i]

    def to_numpy(self, zero_copy_only=True):
        storage: pa.ListArray = self.storage
        null_mask = storage.is_null().to_numpy(zero_copy_only=False)

        if self.type.shape[0] is not None:
            size = 1
            null_indices = np.arange(len(storage))[null_mask] - np.arange(
                np.sum(null_mask)
            )

            for i in range(self.type.ndims):
                size *= self.type.shape[i]
                storage = storage.flatten()
            numpy_arr = storage.to_numpy(zero_copy_only=zero_copy_only)
            numpy_arr = numpy_arr.reshape(
                len(self) - len(null_indices), *self.type.shape
            )

            if len(null_indices):
                numpy_arr = np.insert(
                    numpy_arr.astype(np.float64), null_indices, np.nan, axis=0
                )

        else:
            shape = self.type.shape
            ndims = self.type.ndims
            arrays = []
            first_dim_offsets = np.array([off.as_py() for off in storage.offsets])
            for i, is_null in enumerate(null_mask):
                if is_null:
                    arrays.append(np.nan)
                else:
                    storage_el = storage[i : i + 1]
                    first_dim = first_dim_offsets[i + 1] - first_dim_offsets[i]
                    # flatten storage
                    for _ in range(ndims):
                        storage_el = storage_el.flatten()

                    numpy_arr = storage_el.to_numpy(zero_copy_only=zero_copy_only)
                    arrays.append(numpy_arr.reshape(first_dim, *shape[1:]))

            if len(np.unique(np.diff(first_dim_offsets))) > 1:
                # ragged
                numpy_arr = np.empty(len(arrays), dtype=object)
                numpy_arr[:] = arrays
            else:
                numpy_arr = np.array(arrays)

        return numpy_arr

    def to_pylist(self):
        zero_copy_only = _is_zero_copy_only(self.storage.type, unnest=True)
        numpy_arr = self.to_numpy(zero_copy_only=zero_copy_only)
        if self.type.shape[0] is None and numpy_arr.dtype == object:
            return [arr.tolist() for arr in numpy_arr.tolist()]
        else:
            return numpy_arr.tolist()


class PandasArrayExtensionDtype(PandasExtensionDtype):
    _metadata = "value_type"

    def __init__(self, value_type: Union["PandasArrayExtensionDtype", np.dtype]):
        self._value_type = value_type

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]):
        if isinstance(array, pa.ChunkedArray):
            array = array.type.wrap_array(
                pa.concat_arrays([chunk.storage for chunk in array.chunks])
            )
        zero_copy_only = _is_zero_copy_only(array.storage.type, unnest=True)
        numpy_arr = array.to_numpy(zero_copy_only=zero_copy_only)
        return PandasArrayExtensionArray(numpy_arr)

    @classmethod
    def construct_array_type(cls):
        return PandasArrayExtensionArray

    @property
    def type(self) -> type:
        return np.ndarray

    @property
    def kind(self) -> str:
        return "O"

    @property
    def name(self) -> str:
        return f"array[{self.value_type}]"

    @property
    def value_type(self) -> np.dtype:
        return self._value_type


class PandasArrayExtensionArray(PandasExtensionArray):
    def __init__(self, data: np.ndarray, copy: bool = False):
        self._data = data if not copy else np.array(data)
        self._dtype = PandasArrayExtensionDtype(data.dtype)

    def __array__(self, dtype=None):
        """
        Convert to NumPy Array.
        Note that Pandas expects a 1D array when dtype is set to object.
        But for other dtypes, the returned shape is the same as the one of ``data``.

        More info about pandas 1D requirement for PandasExtensionArray here:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionArray.html#pandas.api.extensions.ExtensionArray

        """
        if dtype == np.dtype(object):
            out = np.empty(len(self._data), dtype=object)
            for i in range(len(self._data)):
                out[i] = self._data[i]
            return out
        if dtype is None:
            return self._data
        else:
            return self._data.astype(dtype)

    def copy(self, deep: bool = False) -> "PandasArrayExtensionArray":
        return PandasArrayExtensionArray(self._data, copy=True)

    @classmethod
    def _from_sequence(
        cls,
        scalars,
        dtype: Optional[PandasArrayExtensionDtype] = None,
        copy: bool = False,
    ) -> "PandasArrayExtensionArray":
        if len(scalars) > 1 and all(
            isinstance(x, np.ndarray)
            and x.shape == scalars[0].shape
            and x.dtype == scalars[0].dtype
            for x in scalars
        ):
            data = np.array(
                scalars, dtype=dtype if dtype is None else dtype.value_type, copy=copy
            )
        else:
            data = np.empty(len(scalars), dtype=object)
            data[:] = scalars
        return cls(data, copy=copy)

    @classmethod
    def _concat_same_type(
        cls, to_concat: SequenceCollections["PandasArrayExtensionArray"]
    ) -> "PandasArrayExtensionArray":
        if len(to_concat) > 1 and all(
            va._data.shape == to_concat[0]._data.shape
            and va._data.dtype == to_concat[0]._data.dtype
            for va in to_concat
        ):
            data = np.vstack([va._data for va in to_concat])
        else:
            data = np.empty(len(to_concat), dtype=object)
            data[:] = [va._data for va in to_concat]
        return cls(data, copy=False)

    @property
    def dtype(self) -> PandasArrayExtensionDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def isna(self) -> np.ndarray:
        return np.array([pd.isna(arr).any() for arr in self._data])

    def __setitem__(self, key: Union[int, slice, np.ndarray], value: Any) -> None:
        raise NotImplementedError()

    def __getitem__(
        self, item: Union[int, slice, np.ndarray]
    ) -> Union[np.ndarray, "PandasArrayExtensionArray"]:
        if isinstance(item, int):
            return self._data[item]
        return PandasArrayExtensionArray(self._data[item], copy=False)

    def take(
        self,
        indices: SequenceCollections[int],
        allow_fill: bool = False,
        fill_value: bool = None,
    ) -> "PandasArrayExtensionArray":
        indices: np.ndarray = np.asarray(indices, dtype=int)
        if allow_fill:
            fill_value = (
                self.dtype.na_value
                if fill_value is None
                else np.asarray(fill_value, dtype=self.dtype.value_type)
            )
            mask = indices == -1
            if (indices < -1).any():
                raise ValueError(
                    "Invalid value in `indices`, must be all >= -1 for `allow_fill` is True"
                )
            elif len(self) > 0:
                pass
            elif not np.all(mask):
                raise IndexError(
                    "Invalid take for empty PandasArrayExtensionArray, must be all -1."
                )
            else:
                data = np.array(
                    [fill_value] * len(indices), dtype=self.dtype.value_type
                )
                return PandasArrayExtensionArray(data, copy=False)
        took = self._data.take(indices, axis=0)
        if allow_fill and mask.any():
            took[mask] = [fill_value] * np.sum(mask)
        return PandasArrayExtensionArray(took, copy=False)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:
        if not isinstance(other, PandasArrayExtensionArray):
            raise NotImplementedError(f"Invalid type to compare to: {type(other)}")
        return (self._data == other._data).all()


def pandas_types_mapper(dtype):
    if isinstance(dtype, _ArrayXDExtensionType):
        return PandasArrayExtensionDtype(dtype.value_type)


class _ArrayXD:
    def __post_init__(self):
        self.shape = tuple(self.shape)

    def __call__(self):
        pa_type = globals()[self.__class__.__name__ + "ExtensionType"](
            self.shape, self.dtype
        )
        return pa_type

    def encode(self, value):
        return value


@dataclass
class Array2D(_ArrayXD):
    """Create a two-dimensional array.

    Args:
        shape (`tuple`):
            Size of each dimension.
        dtype (`str`):
            Name of the data type.

    """

    shape: tuple
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    _type: str = field(default="Array2D", init=False, repr=False)


@dataclass
class Array3D(_ArrayXD):
    """Create a three-dimensional array.

    Args:
        shape (`tuple`):
            Size of each dimension.
        dtype (`str`):
            Name of the data type.

    """

    shape: tuple
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    _type: str = field(default="Array3D", init=False, repr=False)


@dataclass
class Array4D(_ArrayXD):
    """Create a four-dimensional array.

    Args:
        shape (`tuple`):
            Size of each dimension.
        dtype (`str`):
            Name of the data type.

    """

    shape: tuple
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    _type: str = field(default="Array4D", init=False, repr=False)


@dataclass
class Array5D(_ArrayXD):
    """Create a five-dimensional array.

    Args:
        shape (tuple): Size of each dimension.
        dtype (str): Name of the data type.

    """

    shape: tuple
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    _type: str = field(default="Array5D", init=False, repr=False)


class _ArrayXDExtensionType(pa.ExtensionType):
    ndims: Optional[int] = None

    def __init__(self, shape: tuple, dtype: str):
        if self.ndims is None or self.ndims <= 1:
            raise ValueError(
                "You must instantiate an array type with a value for dim that is > 1"
            )
        if len(shape) != self.ndims:
            raise ValueError(f"shape={shape} and ndims={self.ndims} don't match")
        for dim in range(1, self.ndims):
            if shape[dim] is None:
                raise ValueError(
                    f"Support only dynamic size on first dimension. Got: {shape}"
                )
        self.shape = tuple(shape)
        self.value_type = dtype
        self.storage_dtype = self._generate_dtype(self.value_type)
        pa.ExtensionType.__init__(
            self,
            self.storage_dtype,
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )

    def __arrow_ext_serialize__(self):
        return json.dumps((self.shape, self.value_type)).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        args = json.loads(serialized)
        return cls(*args)

    # This was added to pa.ExtensionType in pyarrow >= 13.0.0
    def __reduce__(self):
        return self.__arrow_ext_deserialize__, (
            self.storage_type,
            self.__arrow_ext_serialize__(),
        )

    def __hash__(self):
        return hash((self.__class__, self.shape, self.value_type))

    def __arrow_ext_class__(self):
        return ArrayExtensionArray

    def _generate_dtype(self, dtype):
        dtype = str_to_arrow_type(dtype)
        for d in reversed(self.shape):
            dtype = pa.list_(dtype)
            # Don't specify the size of the list, since fixed length list arrays have issues
            # being validated after slicing in pyarrow 0.17.1
        return dtype

    def to_pandas_dtype(self):
        return PandasArrayExtensionDtype(self.value_type)


class Array2DExtensionType(_ArrayXDExtensionType):
    ndims = 2


class Array3DExtensionType(_ArrayXDExtensionType):
    ndims = 3


class Array4DExtensionType(_ArrayXDExtensionType):
    ndims = 4


class Array5DExtensionType(_ArrayXDExtensionType):
    ndims = 5
