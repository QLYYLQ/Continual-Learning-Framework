import copy
from typing import Union, TypeVar, TYPE_CHECKING

import numpy as np
import pyarrow as pa

from CLTrainingFramework.dataset.arrow_handler.arrow_table.block_table import Table, BlockTable, MemoryTable
from CLTrainingFramework.dataset.arrow_handler.arrow_table.utils import table_flatten, table_cast, concat_for_pyarrow, \
    pyarrow_table_from_grid, merge_specific_table_type_from_blocks
from CLTrainingFramework.dataset.arrow_utils import short_str, wrap_for_chunked_arrays

if TYPE_CHECKING:
    from CLTrainingFramework.dataset.schema import SchemaType

_T_Table = TypeVar("_T_Table", BlockTable, list[BlockTable], list[list[BlockTable]])


class ConcatenationTable(Table):

    def __init__(self, table: pa.Table, blocks: list[list[BlockTable]]):
        super().__init__(table)
        self.blocks = blocks
        # Check that all the blocks have the right type.
        # Only InMemoryTable and MemoryMappedTable are allowed.
        for sub_tables in blocks:
            for sub_table in sub_tables:
                if not isinstance(sub_table, BlockTable):
                    raise TypeError(
                        "The blocks of a ConcatenationTable must be MemoryTable or MemoryMappedTable objects"
                        f", but got {short_str(sub_table)}."
                    )

    def __getstate__(self):
        return {"blocks": self.blocks, "schema": self.table.schema}

    def __setstate__(self, state):
        blocks = state["blocks"]
        schema = state["schema"]
        table = pyarrow_table_from_grid(blocks)
        if schema is not None and table.schema != schema:
            # We fix the columns by concatenating with an empty table with the right columns
            empty_table = pa.Table.from_batches([], schema=schema)
            # We set promote_options="default" to fill missing columns with null values
            table = pa.concat_tables([table, empty_table], promote_options="default")
        ConcatenationTable.__init__(self, table, blocks=blocks)

    @classmethod
    def _consolidate_blocks(cls, blocks: _T_Table) -> _T_Table:
        if isinstance(blocks, BlockTable):
            return blocks
        elif isinstance(blocks[0], BlockTable):
            return merge_specific_table_type_from_blocks(MemoryTable, blocks, axis=0)
        else:
            return merge_specific_table_type_from_blocks(MemoryTable, blocks)

    @classmethod
    def from_blocks(cls, blocks: _T_Table) -> "ConcatenationTable":
        """
        blocks: MemoryTable, BlockTable, list[BlockTable], list[list[BlockTable]]
            if MemoryTable
                blocks == [[blocks]]
            if list[MemoryTable]
                看merge_specific_table_type_from_blocks
            if list[list[BlockTable]]
                看merge_specific_table_type_from_blocks
                然后 pyarrow_table_from_grid
        """
        blocks = cls._consolidate_blocks(blocks)
        if isinstance(blocks, BlockTable):
            table = blocks
            return cls(table.table, [[table]])
        elif isinstance(blocks[0], BlockTable):
            table = concat_for_pyarrow(blocks, axis=0)
            blocks = [[t] for t in blocks]
            return cls(table, blocks)
        else:
            table = pyarrow_table_from_grid(blocks)
            return cls(table, blocks)

    @classmethod
    def from_tables(cls, tables: list[Union[pa.Table, Table]], axis: int = 0) -> "ConcatenationTable":
        """Create `ConcatenationTable` from list of tables.

        Args:
            tables (list of `Table` or list of `pyarrow.Table`):
                List of tables.
            axis (`{0, 1}`, defaults to `0`, meaning over rows):
                Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns
                (horizontally).

                <Added version="1.6.0"/>
        """

        def to_blocks(table: Union[pa.Table, Table]) -> list[list[BlockTable]]:
            if isinstance(table, pa.Table):
                return [[MemoryTable(table)]]
            elif isinstance(table, ConcatenationTable):
                return copy.deepcopy(table.blocks)
            else:
                return [[table]]

        def _slice_row_block(row_block: list[BlockTable], length: int) -> tuple[list[BlockTable], list[BlockTable]]:
            sliced = [table.slice(0, length) for table in row_block]
            remainder = [table.slice(length, len(row_block[0]) - length) for table in row_block]
            return sliced, remainder

        def _split_both_like(
                result: list[list[BlockTable]], blocks: list[list[BlockTable]]
        ) -> tuple[list[list[BlockTable]], list[list[BlockTable]]]:
            """
            Make sure each row_block contain the same num_rows to be able to concatenate them on axis=1.

            To do so, we modify both blocks sets to have the same row_blocks boundaries.
            For example, if `result` has 2 row_blocks of 3 rows and `blocks` has 3 row_blocks of 2 rows,
            we modify both to have 4 row_blocks of size 2, 1, 1 and 2:

                    [ x   x   x | x   x   x ]
                +   [ y   y | y   y | y   y ]
                -----------------------------
                =   [ x   x | x | x | x   x ]
                    [ y   y | y | y | y   y ]

            """
            result, blocks = list(result), list(blocks)
            new_result, new_blocks = [], []
            while result and blocks:
                # we slice the longest row block to save two row blocks of same length
                # and we replace the long row block by its remainder if necessary
                if len(result[0][0]) > len(blocks[0][0]):
                    new_blocks.append(blocks[0])
                    sliced, result[0] = _slice_row_block(result[0], len(blocks.pop(0)[0]))
                    new_result.append(sliced)
                elif len(result[0][0]) < len(blocks[0][0]):
                    new_result.append(result[0])
                    sliced, blocks[0] = _slice_row_block(blocks[0], len(result.pop(0)[0]))
                    new_blocks.append(sliced)
                else:
                    new_result.append(result.pop(0))
                    new_blocks.append(blocks.pop(0))
            if result or blocks:
                raise ValueError("Failed to concatenate on axis=1 because tables don't have the same number of rows")
            return new_result, new_blocks

        def _extend_blocks(
                result: list[list[BlockTable]], blocks: list[list[BlockTable]], axis: int = 0
        ) -> list[list[BlockTable]]:
            if axis == 0:
                result.extend(blocks)
            elif axis == 1:
                # We make sure each row_block have the same num_rows
                result, blocks = _split_both_like(result, blocks)
                for i, row_block in enumerate(blocks):
                    result[i].extend(row_block)
            return result

        blocks = to_blocks(tables[0])
        for table in tables[1:]:
            table_blocks = to_blocks(table)
            blocks = _extend_blocks(blocks, table_blocks, axis=axis)
        return cls.from_blocks(blocks)

    @property
    def _slices(self):
        offset = 0
        for tables in self.blocks:
            length = len(tables[0])
            yield (offset, length)
            offset += length

    def slice(self, offset=0, length=None):
        """
        Compute zero-copy slice of this Table.

        Args:
            offset (`int`, defaults to `0`):
                Offset from start of table to slice.
            length (`int`, defaults to `None`):
                Length of slice (default is until end of table starting from
                offset).

        Returns:
            `datasets.table.Table`
        """
        table = self.table.slice(offset, length=length)
        length = length if length is not None else self.num_rows - offset
        blocks = []
        for tables in self.blocks:
            n_rows = len(tables[0])
            if length == 0:
                break
            elif n_rows <= offset:
                offset = offset - n_rows
            elif n_rows <= offset + length:
                blocks.append([t.slice(offset) for t in tables])
                length, offset = length + offset - n_rows, 0
            else:
                blocks.append([t.slice(offset, length) for t in tables])
                length, offset = 0, 0
        return ConcatenationTable(table, blocks)

    def filter(self, mask, *args, **kwargs):
        """
        Select records from a Table. See `pyarrow.compute.filter` for full usage.
        """
        table = self.table.filter(mask, *args, **kwargs)
        blocks = []
        for (offset, length), tables in zip(self._slices, self.blocks):
            submask = mask.slice(offset, length)
            blocks.append([t.filter(submask, *args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def flatten(self, *args, **kwargs):
        """
        Flatten this Table.  Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        table = table_flatten(self.table, *args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.flatten(*args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def combine_chunks(self, *args, **kwargs):
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the `ChunkedArray` of each column are
        concatenated into zero or one chunk.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        table = self.table.combine_chunks(*args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.combine_chunks(*args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def cast(self, target_schema, *args, **kwargs):
        """
        Cast table values to another schema.

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        from CLTrainingFramework.dataset.schema import Schema

        table = table_cast(self.table, target_schema, *args, **kwargs)
        target_features = Schema.from_arrow_schema(target_schema)
        blocks = []
        for subtables in self.blocks:
            new_tables = []
            fields = list(target_schema)
            for subtable in subtables:
                subfields = []
                for name in subtable.column_names:
                    subfields.append(fields.pop(next(i for i, field in enumerate(fields) if field.name == name)))
                subfeatures = Schema({subfield.name: target_features[subfield.name] for subfield in subfields})
                subschema = subfeatures.to_arrow_schema()
                new_tables.append(subtable.cast(subschema, *args, **kwargs))
            blocks.append(new_tables)
        return ConcatenationTable(table, blocks)

    def replace_schema_metadata(self, *args, **kwargs):
        """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be `None`,
        which deletes any existing metadata).

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
        table = self.table.replace_schema_metadata(*args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.replace_schema_metadata(*args, **kwargs) for t in tables])
        return ConcatenationTable(table, self.blocks)

    def add_column(self, *args, **kwargs):
        """
        没做
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column added.
        """
        raise NotImplementedError("好哥哥，还没做")

    def append_column(self, *args, **kwargs):
        """
        Append column at end of columns.

        Args:
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column added.
        """
        raise NotImplementedError("好哥哥，还没做")

    def remove_column(self, i, *args, **kwargs):
        """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`:
                New table without the column.
        """
        table = self.table.remove_column(i, *args, **kwargs)
        name = self.table.column_names[i]
        blocks = []
        for tables in self.blocks:
            blocks.append(
                [
                    t.remove_column(t.column_names.index(name), *args, **kwargs) if name in t.column_names else t
                    for t in tables
                ]
            )
        return ConcatenationTable(table, blocks)

    def set_column(self, *args, **kwargs):
        """
        Replace column in Table at position.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column set.
        """
        raise NotImplementedError()

    def rename_columns(self, names, *args, **kwargs):
        """
        Create new table with columns renamed to provided names.
        """
        table = self.table.rename_columns(names, *args, **kwargs)
        names = dict(zip(self.table.column_names, names))
        blocks = []
        for tables in self.blocks:
            blocks.append(
                [t.rename_columns([names[name] for name in t.column_names], *args, **kwargs) for t in tables]
            )
        return ConcatenationTable(table, blocks)

    def drop(self, columns, *args, **kwargs):
        """
        Drop one or more columns and return a new table.

        Args:
            columns (`List[str]`):
                List of field names referencing existing columns.

        Raises:
            `KeyError` : if any of the passed columns name are not existing.

        Returns:
            `datasets.table.Table`:
                New table without the columns.
        """
        table = self.table.drop(columns, *args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.drop([c for c in columns if c in t.column_names], *args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)

    def select(self, columns, *args, **kwargs):
        """
        Select columns of the table.

        Returns a new table with the specified columns, and metadata preserved.

        Args:
            columns (:obj:`Union[List[str], List[int]]`):
                The column names or integer indices to select.

        Returns:
            :class:`datasets.table.Table`: New table with the specified columns, and metadata preserved.
        """
        table = self.table.select(columns, *args, **kwargs)
        blocks = []
        for tables in self.blocks:
            blocks.append([t.select([c for c in columns if c in t.column_names], *args, **kwargs) for t in tables])
        return ConcatenationTable(table, blocks)


def concat_tables(tables: list[Table], axis: int = 0) -> Table:
    """
    Concatenate tables.

    Args:
        tables (list of `Table`):
            List of tables to be concatenated.
        axis (`{0, 1}`, defaults to `0`, meaning over rows):
            Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns
            (horizontally).
    """
    tables = list(tables)
    if len(tables) == 1:
        return tables[0]
    return ConcatenationTable.from_tables(tables, axis=axis)
@wrap_for_chunked_arrays
def for_storage(array:pa.Array,schema:"SchemaType"):
    from CLTrainingFramework.dataset.schema import Sequence
    _e = for_storage


if __name__ == '__main__':
    array = np.array([100, 200, 304, 443, 561])
    a = np.searchsorted(array, 200, side="left")
    print(a, type(a))
