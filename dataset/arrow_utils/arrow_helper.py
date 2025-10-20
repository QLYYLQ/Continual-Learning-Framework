import numpy as np
from typing import Union, Any
from functools import partial
import pyarrow as pa
import re
import pyarrow.compute as pc


def arrow_type_to_framework_string_dtype(arrow_type: pa.DataType) -> str:
    """
    arrow_type_to_framework_string_dtype takes a pyarrow.DataType and converts it to a framework string dtype.
    """
    if pa.types.is_null(arrow_type):
        return "null"
    elif pa.types.is_boolean(arrow_type):
        return "bool"
    elif pa.types.is_int8(arrow_type):
        return "int8"
    elif pa.types.is_int16(arrow_type):
        return "int16"
    elif pa.types.is_int32(arrow_type):
        return "int32"
    elif pa.types.is_int64(arrow_type):
        return "int64"
    elif pa.types.is_uint8(arrow_type):
        return "uint8"
    elif pa.types.is_uint16(arrow_type):
        return "uint16"
    elif pa.types.is_uint32(arrow_type):
        return "uint32"
    elif pa.types.is_uint64(arrow_type):
        return "uint64"
    elif pa.types.is_float16(arrow_type):
        return "float16"  # pa dtype is "halffloat"
    elif pa.types.is_float32(arrow_type):
        return "float32"  # pa dtype is "float"
    elif pa.types.is_float64(arrow_type):
        return "float64"  # pa dtype is "double"
    elif pa.types.is_time32(arrow_type):
        return f"time32[{pa.type_for_alias(str(arrow_type)).unit}]"
    elif pa.types.is_time64(arrow_type):
        return f"time64[{pa.type_for_alias(str(arrow_type)).unit}]"
    elif pa.types.is_timestamp(arrow_type):
        if arrow_type.tz is None:
            return f"timestamp[{arrow_type.unit}]"
        elif arrow_type.tz:
            return f"timestamp[{arrow_type.unit}, tz={arrow_type.tz}]"
        else:
            raise ValueError(f"Unexpected timestamp object {arrow_type}.")
    elif pa.types.is_date32(arrow_type):
        return "date32"  # pa dtype is "date32[day]"
    elif pa.types.is_date64(arrow_type):
        return "date64"  # pa dtype is "date64[ms]"
    elif pa.types.is_duration(arrow_type):
        return f"duration[{arrow_type.unit}]"
    elif pa.types.is_decimal128(arrow_type):
        return f"decimal128({arrow_type.precision}, {arrow_type.scale})"
    elif pa.types.is_decimal256(arrow_type):
        return f"decimal256({arrow_type.precision}, {arrow_type.scale})"
    elif pa.types.is_binary(arrow_type):
        return "binary"
    elif pa.types.is_large_binary(arrow_type):
        return "large_binary"
    elif pa.types.is_string(arrow_type):
        return "string"
    elif pa.types.is_large_string(arrow_type):
        return "large_string"
    elif pa.types.is_dictionary(arrow_type):
        return arrow_type_to_framework_string_dtype(arrow_type.value_type)
    else:
        raise ValueError(
            f"Arrow type {arrow_type} does not have a datasets dtype equivalent."
        )


def str_to_arrow_type(framework_dtype: str) -> pa.DataType:
    """

    string_to_arrow takes a datasets string dtype and converts it to a pyarrow.DataType.

    >>>a = str_to_arrow_type( arrow_type_to_framework_string_dtype(a) )

    This is necessary because the CLTrainingFramework.dataset.Value() primitive type is constructed using a string dtype

    Value(dtype=str)

    But Features.type (via `get_nested_type()`) expects to resolve Features into a pyarrow Schema,
        which means that each Value() must be able to resolve into a corresponding pyarrow.DataType, which is the
        purpose of this function.
    """

    def _dtype_error_msg(dtype, pa_dtype, examples=None, urls=None):
        msg = f"{dtype} is not a validly formatted string representation of the pyarrow {pa_dtype} type."
        if examples:
            examples = (
                ", ".join(examples[:-1]) + " or " + examples[-1]
                if len(examples) > 1
                else examples[0]
            )
            msg += f"\nValid examples include: {examples}."
        if urls:
            urls = (
                ", ".join(urls[:-1]) + " and " + urls[-1] if len(urls) > 1 else urls[0]
            )
            msg += f"\nFor more insformation, see: {urls}."
        return msg

    # 直接check dict 中有的方法
    if framework_dtype in pa.__dict__:
        return pa.__dict__[framework_dtype]()

    if (framework_dtype + "_") in pa.__dict__:
        return pa.__dict__[framework_dtype + "_"]()
    # copy from datasets, for timestamp, duration... any of time type in pyarrow😓
    timestamp_matches = re.search(r"^timestamp\[(.*)\]$", framework_dtype)
    if timestamp_matches:
        timestamp_internals = timestamp_matches.group(1)
        internals_matches = re.search(
            r"^(s|ms|us|ns),\s*tz=([a-zA-Z0-9/_+\-:]*)$", timestamp_internals
        )
        if timestamp_internals in ["s", "ms", "us", "ns"]:
            return pa.timestamp(timestamp_internals)
        elif internals_matches:
            return pa.timestamp(internals_matches.group(1), internals_matches.group(2))
        else:
            raise ValueError(
                _dtype_error_msg(
                    framework_dtype,
                    "timestamp",
                    examples=["timestamp[us]", "timestamp[us, tz=America/New_York"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.timestamp.html"
                    ],
                )
            )

    duration_matches = re.search(r"^duration\[(.*)\]$", framework_dtype)
    if duration_matches:
        duration_internals = duration_matches.group(1)
        if duration_internals in ["s", "ms", "us", "ns"]:
            return pa.duration(duration_internals)
        else:
            raise ValueError(
                _dtype_error_msg(
                    framework_dtype,
                    "duration",
                    examples=["duration[s]", "duration[us]"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.duration.html"
                    ],
                )
            )

    time_matches = re.search(r"^time(.*)\[(.*)\]$", framework_dtype)
    if time_matches:
        time_internals_bits = time_matches.group(1)
        if time_internals_bits == "32":
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ["s", "ms"]:
                return pa.time32(time_internals_unit)
            else:
                raise ValueError(
                    f"{time_internals_unit} is not a valid unit for the pyarrow time32 type. Supported units: s (second) and ms (millisecond)."
                )
        elif time_internals_bits == "64":
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ["us", "ns"]:
                return pa.time64(time_internals_unit)
            else:
                raise ValueError(
                    f"{time_internals_unit} is not a valid unit for the pyarrow time64 type. Supported units: us (microsecond) and ns (nanosecond)."
                )
        else:
            raise ValueError(
                _dtype_error_msg(
                    framework_dtype,
                    "time",
                    examples=["time32[s]", "time64[us]"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.time32.html",
                        "https://arrow.apache.org/docs/python/generated/pyarrow.time64.html",
                    ],
                )
            )

    decimal_matches = re.search(r"^decimal(.*)\((.*)\)$", framework_dtype)
    if decimal_matches:
        decimal_internals_bits = decimal_matches.group(1)
        if decimal_internals_bits == "128":
            decimal_internals_precision_and_scale = re.search(
                r"^(\d+),\s*(-?\d+)$", decimal_matches.group(2)
            )
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal128(int(precision), int(scale))
            else:
                raise ValueError(
                    _dtype_error_msg(
                        framework_dtype,
                        "decimal128",
                        examples=["decimal128(10, 2)", "decimal128(4, -2)"],
                        urls=[
                            "https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html"
                        ],
                    )
                )
        elif decimal_internals_bits == "256":
            decimal_internals_precision_and_scale = re.search(
                r"^(\d+),\s*(-?\d+)$", decimal_matches.group(2)
            )
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal256(int(precision), int(scale))
            else:
                raise ValueError(
                    _dtype_error_msg(
                        framework_dtype,
                        "decimal256",
                        examples=["decimal256(30, 2)", "decimal256(38, -4)"],
                        urls=[
                            "https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html"
                        ],
                    )
                )
        else:
            raise ValueError(
                _dtype_error_msg(
                    framework_dtype,
                    "decimal",
                    examples=["decimal128(12, 3)", "decimal256(40, 6)"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html",
                        "https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html",
                    ],
                )
            )

    raise ValueError(
        f"Neither {framework_dtype} nor {framework_dtype + '_'} seems to be a pyarrow data type. "
        f"Please make sure to use a correct data type, see: "
        f"https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions"
    )


def _is_zero_copy_only(pa_type: pa.DataType, unnest: bool = False) -> bool:
    """
    When converting a pyarrow array to a numpy array, we must know whether this could be done in zero-copy or not.
    This function returns the value of the ``zero_copy_only`` parameter to pass to ``.to_numpy()``, given the type of the pyarrow array.

    https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.to_numpy
    https://arrow.apache.org/docs/python/generated/pyarrow.types.is_primitive.html

    """

    def _unnest_pa_type(pa_type: pa.DataType) -> pa.DataType:
        if pa.types.is_list(pa_type):
            return _unnest_pa_type(pa_type.value_type)
        return pa_type

    if unnest:
        pa_type = _unnest_pa_type(pa_type)
    return pa.types.is_primitive(pa_type) and not (
            pa.types.is_boolean(pa_type) or pa.types.is_temporal(pa_type)
    )


def wrap_for_chunked_arrays(func):
    """Apply the function on each chunk of a `pyarrow.ChunkedArray`, or on the array directly"""

    def wrapper(array, *args, **kwargs):
        if isinstance(array, pa.ChunkedArray):
            return pa.chunked_array(
                [func(chunk, *args, **kwargs) for chunk in array.chunks]
            )
        else:
            return func(array, *args, **kwargs)

    return wrapper


def storage_type(type: pa.DataType) -> pa.DataType:
    """Convert a (possibly nested) `pa.ExtensionType` to its storage type."""
    if isinstance(type, pa.ExtensionType):
        return storage_type(type.storage_type)
    elif isinstance(type, pa.StructType):
        return pa.struct(
            [pa.field(field.name, storage_type(field.type)) for field in type]
        )
    elif isinstance(type, pa.ListType):
        return pa.list_(storage_type(type.value_type))
    elif isinstance(type, pa.FixedSizeListType):
        return pa.list_(storage_type(type.value_type), type.list_size)
    return type


def combine_list_array_offsets_with_mask(array: pa.ListArray) -> pa.Array:
    """Add the null bitmap to the offsets of a `pa.ListArray`."""
    offsets = array.offsets
    if array.null_count > 0:
        offsets = pa.concat_arrays(
            [
                pc.replace_with_mask(
                    offsets[:-1], array.is_null(), pa.nulls(len(array), pa.int32())
                ),
                offsets[-1:],
            ]
        )
    return offsets


def short_str(value: Any) -> str:
    out = str(value)
    if len(out) > 3000:
        out = out[:1500] + "\n...\n" + out[-1500:]
    return out


def check_sub_list_length_for_ListArray(array: pa.ListArray, length: int) -> bool:
    """Check if all the sub-lists of a `pa.ListArray` have the specified length."""
    return pc.all(
        pc.equal(array.value_lengths(), length)
    ).as_py() or array.null_count == len(array)


@wrap_for_chunked_arrays
def array_cast(
        array: pa.Array,
        pa_type: pa.DataType,
        allow_primitive_to_str: bool = True,
        allow_decimal_to_str: bool = True,
) -> Union[
    pa.Array, pa.FixedSizeListArray, pa.ListArray, pa.StructArray, pa.ExtensionArray
]:
    """
    From huggingface

    Improved version of `pa.Array.cast`

    It supports casting `pa.StructArray` objects to re-order the fields.
    It also let you control certain aspects of the casting, e.g. whether
    to disable casting primitives (`booleans`, `floats` or `ints`) or
    disable casting decimals to strings.

    Args:
        array (`pa.Array`):
            PyArrow array to cast
        pa_type (`pa.DataType`):
            Target PyArrow type
        allow_primitive_to_str (`bool`, defaults to `True`):
            Whether to allow casting primitives to strings.
            Defaults to `True`.
        allow_decimal_to_str (`bool`, defaults to `True`):
            Whether to allow casting decimals to strings.
            Defaults to `True`.

    Raises:
        `pa.ArrowInvalidError`: if the arrow data casting fails
        `TypeError`: if the target type is not supported according, e.g.

            - if a field is missing
            - if casting from primitives to strings and `allow_primitive_to_str` is `False`
            - if casting from decimals to strings and `allow_decimal_to_str` is `False`

    Returns:
        `List[pyarrow.Array]`: the casted array
    """
    _c = partial(
        array_cast,
        allow_primitive_to_str=allow_primitive_to_str,
        allow_decimal_to_str=allow_decimal_to_str,
    )
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if isinstance(pa_type, pa.ExtensionType):
        return pa_type.wrap_array(_c(array, pa_type.storage_type))
    elif array.type == pa_type:
        return array
    elif pa.types.is_struct(array.type):
        if pa.types.is_struct(pa_type) and (
                {field.name for field in pa_type} == {field.name for field in array.type}
        ):
            if array.type.num_fields == 0:
                return array
            arrays = [_c(array.field(field.name), field.type) for field in pa_type]
            return pa.StructArray.from_arrays(
                arrays, fields=list(pa_type), mask=array.is_null()
            )
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        if pa.types.is_fixed_size_list(pa_type):
            if check_sub_list_length_for_ListArray(array, pa_type.list_size):
                if array.null_count > 0:
                    # Ensure each null value in the array translates to [null] * pa_type.list_size in the array's values array
                    array_type = array.type
                    storage_type = storage_type(array_type)
                    if array_type != storage_type:
                        # Temporarily convert to the storage type to support extension types in the slice operation
                        array = _c(array, storage_type)
                        array = pc.list_slice(
                            array, 0, pa_type.list_size, return_fixed_size_list=True
                        )
                        array = _c(array, array_type)
                    else:
                        array = pc.list_slice(
                            array, 0, pa_type.list_size, return_fixed_size_list=True
                        )
                    array_values = array.values
                    return pa.FixedSizeListArray.from_arrays(
                        _c(array_values, pa_type.value_type),
                        pa_type.list_size,
                        mask=array.is_null(),
                    )
                else:
                    array_values = array.values[
                                   array.offset
                                   * pa_type.list_size: (array.offset + len(array))
                                                        * pa_type.list_size
                                   ]
                    return pa.FixedSizeListArray.from_arrays(
                        _c(array_values, pa_type.value_type), pa_type.list_size
                    )
        elif pa.types.is_list(pa_type):
            # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
            array_offsets = combine_list_array_offsets_with_mask(array)
            return pa.ListArray.from_arrays(
                array_offsets, _c(array.values, pa_type.value_type)
            )
        elif pa.types.is_large_list(pa_type):
            # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
            array_offsets = combine_list_array_offsets_with_mask(array)
            return pa.LargeListArray.from_arrays(
                array_offsets, _c(array.values, pa_type.value_type)
            )
    elif pa.types.is_fixed_size_list(array.type):
        if pa.types.is_fixed_size_list(pa_type):
            if pa_type.list_size == array.type.list_size:
                array_values = array.values[
                               array.offset
                               * array.type.list_size: (array.offset + len(array))
                                                       * array.type.list_size
                               ]
                return pa.FixedSizeListArray.from_arrays(
                    _c(array_values, pa_type.value_type),
                    pa_type.list_size,
                    mask=array.is_null(),
                )
        elif pa.types.is_list(pa_type):
            array_offsets = (
                                    np.arange(len(array) + 1) + array.offset
                            ) * array.type.list_size
            return pa.ListArray.from_arrays(
                array_offsets,
                _c(array.values, pa_type.value_type),
                mask=array.is_null(),
            )
        elif pa.types.is_large_list(pa_type):
            array_offsets = (
                                    np.arange(len(array) + 1) + array.offset
                            ) * array.type.list_size
            return pa.LargeListArray.from_arrays(
                array_offsets,
                _c(array.values, pa_type.value_type),
                mask=array.is_null(),
            )
    else:
        if pa.types.is_string(pa_type):
            if not allow_primitive_to_str and pa.types.is_primitive(array.type):
                raise TypeError(
                    f"Couldn't cast array of type {short_str(array.type)} to {short_str(pa_type)} "
                    f"since allow_primitive_to_str is set to {allow_primitive_to_str} "
                )
            if not allow_decimal_to_str and pa.types.is_decimal(array.type):
                raise TypeError(
                    f"Couldn't cast array of type {short_str(array.type)} to {short_str(pa_type)} "
                    f"and allow_decimal_to_str is set to {allow_decimal_to_str}"
                )
        if pa.types.is_null(pa_type) and not pa.types.is_null(array.type):
            raise TypeError(
                f"Couldn't cast array of type {short_str(array.type)} to {short_str(pa_type)}"
            )
        return array.cast(pa_type)
    raise TypeError(
        f"Couldn't cast array of type {short_str(array.type)} to {short_str(pa_type)}"
    )
