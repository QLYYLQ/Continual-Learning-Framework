import copy
from functools import partial
from itertools import groupby
from typing import Any, Union, TYPE_CHECKING, Optional, Iterable, Callable

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

if TYPE_CHECKING:
    from CLTrainingFramework.dataset.schema import SchemaType
    from CLTrainingFramework.dataset.arrow_handler.arrow_table.block_table import BlockTable, MemoryTable,Table
    from CLTrainingFramework.dataset.arrow_handler.arrow_table.table import _T_Table


from CLTrainingFramework.dataset.arrow_handler.arrow_table.error import CastError
from CLTrainingFramework.dataset.arrow_utils import storage_type, short_str, wrap_for_chunked_arrays, \
    check_sub_list_length_for_ListArray, combine_list_array_offsets_with_mask, array_cast


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
def cast_pa_array_using_schema(
        array: pa.Array, _schema: "SchemaType", allow_primitive_to_str: bool = True, allow_decimal_to_str: bool = True
) -> pa.Array:
    """
    """
    from CLTrainingFramework.dataset.schema import LargeSequence, Sequence, schema_to_pyarrow

    _c = partial(
        cast_pa_array_using_schema,
        allow_primitive_to_str=allow_primitive_to_str,
        allow_decimal_to_str=allow_decimal_to_str,
    )

    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    if hasattr(_schema, "prepare_for_pa_cache"):
        return _schema.prepare_for_pa_cache(array)

    elif pa.types.is_struct(array.type):
        # feature must be a dict or Sequence(subfeatures_dict)
        if isinstance(_schema, Sequence) and isinstance(_schema.schema, dict):
            sequence_kwargs = vars(_schema).copy()
            _schema = sequence_kwargs.pop("schema")
            _schema = {name: Sequence(subfeature, **sequence_kwargs) for name, subfeature in _schema.items()}
        if isinstance(_schema, dict) and (array_fields := {field.name for field in array.type}) <= set(_schema):
            null_array = pa.array([None] * len(array))
            arrays = [
                _c(array.field(name) if name in array_fields else null_array, subfeature)
                for name, subfeature in _schema.items()
            ]
            return pa.StructArray.from_arrays(arrays, names=list(_schema), mask=array.is_null())
    elif pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        # feature must be either [subfeature] or LargeList(subfeature) or Sequence(subfeature)
        if isinstance(_schema, list):
            casted_array_values = _c(array.values, _schema[0])
            if pa.types.is_list(array.type) and casted_array_values.type == array.values.type:
                # Both array and feature have equal list type and values (within the list) type
                return array
            else:
                # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
                array_offsets = combine_list_array_offsets_with_mask(array)
                return pa.ListArray.from_arrays(array_offsets, casted_array_values)
        elif isinstance(_schema, LargeSequence):
            casted_array_values = _c(array.values, _schema.schema)
            if pa.types.is_large_list(array.type) and casted_array_values.type == array.values.type:
                # Both array and feature have equal large_list type and values (within the list) type
                return array
            else:
                # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
                array_offsets = combine_list_array_offsets_with_mask(array)
                return pa.LargeListArray.from_arrays(array_offsets, casted_array_values)
        elif isinstance(_schema, Sequence):
            if _schema.length > -1:
                if check_sub_list_length_for_ListArray(array, _schema.length):
                    if array.null_count > 0:
                        # Ensure each null value in the array translates to [null] * pa_type.list_size in the array's values array
                        array_type = array.type
                        _storage_type = storage_type(array_type)
                        if array_type != _storage_type:
                            # Temporarily convert to the storage type to support extension types in the slice operation
                            array = array_cast(
                                array,
                                _storage_type,
                                allow_primitive_to_str=allow_primitive_to_str,
                                allow_decimal_to_str=allow_decimal_to_str,
                            )
                            array = pc.list_slice(array, 0, _schema.length, return_fixed_size_list=True)
                            array = array_cast(
                                array,
                                array_type,
                                allow_primitive_to_str=allow_primitive_to_str,
                                allow_decimal_to_str=allow_decimal_to_str,
                            )
                        else:
                            array = pc.list_slice(array, 0, _schema.length, return_fixed_size_list=True)
                        array_values = array.values
                        casted_array_values = _c(array_values, _schema.schema)
                        return pa.FixedSizeListArray.from_arrays(
                            casted_array_values, _schema.length, mask=array.is_null()
                        )
                    else:
                        array_values = array.values[
                                       array.offset * _schema.length: (array.offset + len(array)) * _schema.length
                                       ]
                        return pa.FixedSizeListArray.from_arrays(_c(array_values, _schema.feature), _schema.length)
            else:
                casted_array_values = _c(array.values, _schema.schema)
                if pa.types.is_list(array.type) and casted_array_values.type == array.values.type:
                    # Both array and feature have equal list type and values (within the list) type
                    return array
                else:
                    # Merge offsets with the null bitmap to avoid the "Null bitmap with offsets slice not supported" ArrowNotImplementedError
                    array_offsets = combine_list_array_offsets_with_mask(array)
                    return pa.ListArray.from_arrays(array_offsets, casted_array_values)
    elif pa.types.is_fixed_size_list(array.type):
        # feature must be either [subfeature] or Sequence(subfeature)
        if isinstance(_schema, list):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.ListArray.from_arrays(array_offsets, _c(array.values, _schema[0]), mask=array.is_null())
        elif isinstance(_schema, LargeSequence):
            array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
            return pa.LargeListArray.from_arrays(
                array_offsets, _c(array.values, _schema.schema), mask=array.is_null()
            )
        elif isinstance(_schema, Sequence):
            if _schema.length > -1:
                if _schema.length == array.type.list_size:
                    array_values = array.values[
                                   array.offset * array.type.list_size: (array.offset + len(
                                       array)) * array.type.list_size
                                   ]
                    casted_array_values = _c(array_values, _schema.schema)
                    return pa.FixedSizeListArray.from_arrays(casted_array_values, _schema.length, mask=array.is_null())
            else:
                array_offsets = (np.arange(len(array) + 1) + array.offset) * array.type.list_size
                return pa.ListArray.from_arrays(array_offsets, _c(array.values, _schema.schema), mask=array.is_null())
    if pa.types.is_null(array.type):
        return array_cast(
            array,
            schema_to_pyarrow(_schema),
            allow_primitive_to_str=allow_primitive_to_str,
            allow_decimal_to_str=allow_decimal_to_str,
        )
    elif not isinstance(_schema, (Sequence, dict, list, tuple)):
        return array_cast(
            array,
            _schema(),
            allow_primitive_to_str=allow_primitive_to_str,
            allow_decimal_to_str=allow_decimal_to_str,
        )
    raise TypeError(f"Couldn't cast array of type\n{short_str(array.type)}\nto\n{short_str(_schema)}")


def cast_pa_table_using_pa_schema(table: pa.Table, schema: pa.Schema):
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
        cast_pa_array_using_schema(
            table[name] if name in table_column_names else pa.array([None] * len(table), type=schema.field(name).type),
            v,
        )
        for name, v in _schema.items()
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def pa_table_cast(table: pa.Table, schema: pa.Schema):
    """

    Improved version of `pa.Table.cast`.

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
        return cast_pa_table_using_pa_schema(table, schema)
    elif table.schema.metadata != schema.metadata:
        return table.replace_schema_metadata(schema.metadata)
    else:
        return table


def _memory_mapped_record_batch_reader_from_file(filename: str) -> pa.RecordBatchStreamReader:
    memory_mapped_stream = pa.memory_map(filename)
    return pa.ipc.open_stream(memory_mapped_stream)


def _memory_mapped_arrow_table_from_file(filename: str) -> pa.Table:
    opened_stream = _memory_mapped_record_batch_reader_from_file(filename)
    pa_table = opened_stream.read_all()
    return pa_table


def concat_for_pyarrow(blocks: list[Union[BlockTable, pa.Table]], axis: int = 0) -> pa.Table:
    """
    类似pandas.concat
    当axis=0的时候，把多个表从上到下堆叠起来，它会以所有表中出现过的列名的并集作为最终表的列
    当axis=1的时候，把多个表从左到右堆叠起来（所有表必须有相同行数）
    """
    pa_tables = [table.table if hasattr(table, "table") else table for table in blocks]
    if axis == 0:
        # We set promote_options="default" to fill missing columns with null values
        return pa.concat_tables(pa_tables, promote_options="default")
    elif axis == 1:
        for i, table in enumerate(pa_tables):
            if i == 0:
                pa_table = table
            else:
                for name, col in zip(table.column_names, table.columns):
                    pa_table = pa_table.append_column(name, col)
        return pa_table
    else:
        raise ValueError("'axis' must be either 0 or 1")


def pyarrow_table_from_grid(blocks: list[list[BlockTable]]) -> pa.Table:
    """
    from blocks like:
        [[Block A, Block B],  // Row 1
        [Block C, Block D] ]  // Row 2
    to one cohesive table:
        +-------+
        | A | B |
        +---+---+
        | C | D |
        +-------+
    It achieves this by first creating the rows [A|B] and [C|D], and then stacking these two rows vertically.
    其中BlockA和BlockB或者BlockC和BlockD必须有相同行数，这两组的列数不必相同
    """
    pa_tables_to_concat_vertically = []
    for i, tables in enumerate(blocks):
        if not tables:
            continue
        pa_table_horizontally_concatenated = concat_for_pyarrow(tables, axis=1)
        pa_tables_to_concat_vertically.append(pa_table_horizontally_concatenated)
    return concat_for_pyarrow(pa_tables_to_concat_vertically, axis=0)


def merge_specific_table_type_from_blocks(table_type: type(MemoryTable), blocks: _T_Table,
                                          axis: Optional[int] = None) -> _T_Table:
    """
    当axis不为None的时候，blocks为list[BlockTable]
        从blocks中，挑选连续的table_type，按照axis进行堆叠，堆叠策略看concat_for_pyarrow的注释
    当axis为None的时候，blocks为list[list[BlockTable]]
        blocks中每一个元素通过concat_for_pyarrow(*,axis=1)进行水平堆叠，再把这些水平堆叠的元素垂直堆一起
    """
    if axis is not None:
        merged_blocks = []
        for is_in_memory, block_group in groupby(blocks, key=lambda x: isinstance(x, table_type)):
            if is_in_memory:
                block_group = [table_type(concat_for_pyarrow(list(block_group), axis=axis))]
            merged_blocks += list(block_group)
    else:  # both
        merged_blocks = [merge_specific_table_type_from_blocks(table_type, row_block, axis=1) for row_block in blocks]
        if all(len(row_block) == 1 for row_block in merged_blocks):
            merged_blocks = merge_specific_table_type_from_blocks(table_type,
                                                                  [block for row_block in merged_blocks for block in
                                                                   row_block], axis=0
                                                                  )
    return merged_blocks

def table_iter(table:Table,batch_size:int,drop_last_batch:bool=False)->Iterable[pa.Table]:
    """
    Args:
        table:
            table to iterate over
        batch_size:
            size of each sub-table to yield
        drop_last_batch:
            drop the last batch if it is smaller than batch_size
    """
    chunks_buffer = [
    ]
    chunks_buffer_size=0
    for chunk in table.to_reader(max_chunk_size=batch_size):
        if len(chunk==0):
            continue
        elif chunks_buffer_size+len(chunk)<batch_size:
            chunks_buffer.append(chunk)
            chunks_buffer_size+=len(chunk)
            continue
        elif chunks_buffer_size+len(chunk)==batch_size:
            chunks_buffer.append(chunk)
            yield pa.Table.from_batches(chunks_buffer)
            chunks_buffer=[]
            chunks_buffer_size=0
        elif chunks_buffer_size+len(chunk)>batch_size:
            cropped_chunk_length = batch_size-chunks_buffer_size
            chunks_buffer.append(chunk.slice(0,cropped_chunk_length))
            yield pa.Table.from_batches(chunks_buffer)
            # 这里是对的，不用担心，api文档里的slice就是这么写的
            chunks_buffer=[chunk.slice(cropped_chunk_length,len(chunk)-cropped_chunk_length)]
            chunks_buffer_size=len(chunk)-cropped_chunk_length
    if not drop_last_batch and chunks_buffer:
        yield pa.Table.from_batches(chunks_buffer)

def map_function_to_table(table:pa.Table,function:Callable[[pa.Array],None]):
    from CLTrainingFramework.dataset.schema import Schema,Sequence
    schema = Schema.from_arrow_schema(table.schema)
    def _visit(array,schema:"SchemaType"):
        if isinstance(array,pa.ChunkedArray):
            for chunk in array.chunks:
                _visit(chunk,schema)
        else:
            if isinstance(array,pa.ExtensionArray):
                array = array.storage
            function(array,schema)
            if pa.types.is_struct(array.type) and not hasattr(schema,"prepare_for_pa_cache"):
                if isinstance(schema,Sequence) and isinstance(schema.schema,dict):
                    schema={
                        k:Sequence(v,length=schema.length)
                        for k,v in schema.schema.items()
                    }
                    for k,v in schema.items():
                        _visit(array.field(k),v)
            elif pa.types.is_list(array.type):
                if isinstance(schema,list):
                    _visit(array.values,schema[0])
                elif isinstance(schema,Sequence):
                    _visit(array.values,schema.schema)
    for k,v in schema.items():
        _visit(table[k],v)
