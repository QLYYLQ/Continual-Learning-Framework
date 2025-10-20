import copy
from functools import partial
from typing import Any, Union, TYPE_CHECKING

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
if TYPE_CHECKING:
    from CLTrainingFramework.dataset.schema import SchemaType

from CLTrainingFramework.dataset.arrow_handler.arrow_table.error import CastError
from CLTrainingFramework.dataset.arrow_utils import storage_type,short_str,wrap_for_chunked_arrays,check_sub_list_length_for_ListArray,combine_list_array_offsets_with_mask,array_cast

def _deepcopy(x, memo: dict):
    """deepcopy a regular class instance"""
    cls = x.__class__
    result = cls.__new__(cls)
    memo[id(x)] = result
    for k, v in x.__dict__.items():
        setattr(result, k, copy.deepcopy(v, memo))
    return result


def _memory_table_from_file(filename: str) -> pa.Table:
    in_memory_stream = pa.input_stream(filename)
    opened_stream = pa.ipc.open_stream(in_memory_stream)
    pa_table = opened_stream.read_all()
    return pa_table


def _memory_table_from_buffer(buffer: pa.Buffer) -> pa.Table:
    stream = pa.BufferReader(buffer)
    opened_stream = pa.ipc.open_stream(stream)
    table = opened_stream.read_all()
    return table


def table_flatten(table: pa.Table):
    """
    Improved version of `pa.Table.flatten`.

    Args:
        table (`pa.Table`):
            PyArrow table to flatten.

    Returns:
        `Table`: the flattened table
    """
    from CLTrainingFramework.dataset.schema import Schema

    _schema = Schema.from_arrow_schema(table.schema)
    if any(hasattr(sub_schema, "flatten") and sub_schema.flatten() == sub_schema for sub_schema in _schema.values()):
        flat_arrays = []
        flat_column_names = []
        for field in table.schema:
            array = table.column(field.name)
            sub_schema = _schema[field.name]
            if pa.types.is_struct(field.type) and (
                    not hasattr(sub_schema, "flatten") or sub_schema.flatten() != sub_schema
            ):
                flat_arrays.extend(array.flatten())
                flat_column_names.extend([f"{field.name}.{subfield.name}" for subfield in field.type])
            else:
                flat_arrays.append(array)
                flat_column_names.append(field.name)
        flat_table = pa.Table.from_arrays(
            flat_arrays,
            names=flat_column_names,
        )
    else:
        flat_table = table.flatten()
    # Preserve complex types in the metadata
    flat_features = _schema.flatten(max_depth=2)
    flat_features = Schema({column_name: flat_features[column_name] for column_name in flat_table.column_names})
    return flat_table.replace_schema_metadata(flat_features.to_arrow_schema().metadata)












@wrap_for_chunked_arrays
def _cast_array_to_schema(
        array: pa.Array, feature: "SchemaType", allow_primitive_to_str: bool = True, allow_decimal_to_str: bool = True
) -> pa.Array:
    """
    """
    from CLTrainingFramework.dataset.schema import LargeSequence, Sequence, schema_to_pyarrow

    _c = partial(
        _cast_array_to_schema,
        allow_primitive_to_str=allow_primitive_to_str,
        allow_decimal_to_str=allow_decimal_to_str,
    )

    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if hasattr(feature, "cast_storage"):
        return feature.cast_storage(array)

    elif pa.types.is_struct(array.type):
        # feature must be a dict or Sequence(subfeatures_dict)
        if isinstance(feature, Sequence) and isinstance(feature.schema, dict):
            sequence_kwargs = vars(feature).copy()
            feature = sequence_kwargs.pop("schema")
            feature = {name: Sequence(subfeature, **sequence_kwargs) for name, subfeature in feature.items()}
        if isinstance(feature, dict) and (array_fields := {field.name for field in array.type}) <= set(feature):
            null_array = pa.array([None] * len(array))
            arrays = [
                _c(array.field(name) if name in array_fields else null_array, subfeature)
                for name, subfeature in feature.items()
            ]
            return pa.StructArray.from_arrays(arrays, names=list(feature), mask=array.is_null())
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        # feature must be either [subfeature] or LargeList(subfeature) or Sequence(subfeature)
        if isinstance(feature, list):
            casted_array_values = _c(array.values, feature[0])
            if pa.types.is_list(array.type) and casted_array_values.type == array.values.type:
                # Both array and feature have equal list type and values (within the list) type
                return array
            else:
                # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
                array_offsets = combine_list_array_offsets_with_mask(array)
                return pa.ListArray.from_arrays(array_offsets, casted_array_values)
        elif isinstance(feature, LargeSequence):
            casted_array_values = _c(array.values, feature.schema)
            if pa.types.is_large_list(array.type) and casted_array_values.type == array.values.type:
                # Both array and feature have equal large_list type and values (within the list) type
                return array
            else:
                # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
                array_offsets = combine_list_array_offsets_with_mask(array)
                return pa.LargeListArray.from_arrays(array_offsets, casted_array_values)
        elif isinstance(feature, Sequence):
            if feature.length > -1:
                if check_sub_list_length_for_ListArray(array, feature.length):
                    if array.null_count > 0:
                        # Ensure each null value in the array translates to [null] * pa_type.list_size in the array's values array
                        array_type = array.type
                        storage_type = storage_type(array_type)
                        if array_type != storage_type:
                            # Temporarily convert to the storage type to support extension types in the slice operation
                            array = array_cast(
                                array,
                                storage_type,
                                allow_primitive_to_str=allow_primitive_to_str,
                                allow_decimal_to_str=allow_decimal_to_str,
                            )
                            array = pc.list_slice(array, 0, feature.length, return_fixed_size_list=True)
                            array = array_cast(
                                array,
                                array_type,
                                allow_primitive_to_str=allow_primitive_to_str,
                                allow_decimal_to_str=allow_decimal_to_str,
                            )
                        else:
                            array = pc.list_slice(array, 0, feature.length, return_fixed_size_list=True)
                        array_values = array.values
                        casted_array_values = _c(array_values, feature.feature)
                        return pa.FixedSizeListArray.from_arrays(
                            casted_array_values, feature.length, mask=array.is_null()
                        )
                    else:
                        array_values = array.values[
                                       array.offset * feature.length: (array.offset + len(array)) * feature.length
                                       ]
                        return pa.FixedSizeListArray.from_arrays(_c(array_values, feature.feature), feature.length)
            else:
                casted_array_values = _c(array.values, feature.schema)
                if pa.types.is_list(array.type) and casted_array_values.type == array.values.type:
                    # Both array and feature have equal list type and values (within the list) type
                    return array
                else:
                    # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
                    array_offsets = combine_list_array_offsets_with_mask(array)
                    return pa.ListArray.from_arrays(array_offsets, casted_array_values)
    elif pa.types.is_fixed_size_list(array.type):
        # feature must be either [subfeature] or Sequence(subfeature)
        if isinstance(feature, list):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.ListArray.from_arrays(array_offsets, _c(array.values, feature[0]), mask=array.is_null())
        elif isinstance(feature, LargeSequence):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.LargeListArray.from_arrays(
                array_offsets, _c(array.values, feature.schema), mask=array.is_null()
            )
        elif isinstance(feature, Sequence):
            if feature.length > -1:
                if feature.length == array.type.list_size:
                    array_values = array.values[
                                   array.offset * array.type.list_size: (array.offset + len(
                                       array)) * array.type.list_size
                                   ]
                    casted_array_values = _c(array_values, feature.schema)
                    return pa.FixedSizeListArray.from_arrays(casted_array_values, feature.length, mask=array.is_null())
            else:
                array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
                return pa.ListArray.from_arrays(array_offsets, _c(array.values, feature.schema), mask=array.is_null())
    if pa.types.is_null(array.type):
        return array_cast(
            array,
            schema_to_pyarrow(feature),
            allow_primitive_to_str=allow_primitive_to_str,
            allow_decimal_to_str=allow_decimal_to_str,
        )
    elif not isinstance(feature, (Sequence, dict, list, tuple)):
        return array_cast(
            array,
            feature(),
            allow_primitive_to_str=allow_primitive_to_str,
            allow_decimal_to_str=allow_decimal_to_str,
        )
    raise TypeError(f"Couldn't cast array of type\n{short_str(array.type)}\nto\n{short_str(feature)}")


def cast_table_to_schema(table: pa.Table, schema: pa.Schema):
    from CLTrainingFramework.dataset.schema import Schema

    _schema = Schema.from_arrow_schema(schema)
    table_column_names = set(table.column_names)
    if not table_column_names <= set(schema.names):
        raise CastError(
            f"Couldn't cast\n{short_str(table.schema)}\nto\n{short_str(_schema)}\nbecause column names don't match",
            table_column_names=table.column_names,
            requested_column_names=list(_schema),
        )
    arrays = [
        _cast_array_to_schema(
            table[name] if name in table_column_names else pa.array([None] * len(table), type=schema.field(name).type),
            v,
        )
        for name, v in _schema.items()
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def table_cast(table: pa.Table, schema: pa.Schema):
    """Improved version of `pa.Table.cast`.

    It supports casting to feature types stored in the schema metadata.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to cast.
        schema (`pyarrow.Schema`):
            Target PyArrow schema.

    Returns:
        table (`pyarrow.Table`): the casted table
    """
    if table.schema != schema:
        return cast_table_to_schema(table, schema)
    elif table.schema.metadata != schema.metadata:
        return table.replace_schema_metadata(schema.metadata)
    else:
        return table
