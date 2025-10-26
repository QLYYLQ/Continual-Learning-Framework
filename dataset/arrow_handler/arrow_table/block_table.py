import copy
import os
from typing import Union, Optional

import numpy as np
import pyarrow as pa

from CLTrainingFramework.dataset.arrow_handler.arrow_table.utils import _deepcopy, _memory_table_from_file, \
    _memory_table_from_buffer, table_flatten, table_cast, _memory_mapped_arrow_table_from_file


class IndexPlugin:
    def __init__(self, table: pa.Table):
        self._schema = table.schema
        self._batches: list[pa.RecordBatch] = [
            i for i in table.to_batches() if len(i) > 0
        ]
        self._offsets: np.array = np.cumsum([0] + [len(i) for i in self._batches], dtype=np.int64)

    def fast_gather(self, indices: Union[list[int], np.array]) -> pa.Table:
        if not len(indices):
            raise ValueError("indices can't be empty")
        batch_indices = np.searchsorted(self._offsets, indices, side="right") - 1
        return pa.Table.from_batches(
            [
                self._batches[j].slice(i - self._offsets[j], 1)
                for j, i in zip(batch_indices, indices)
            ],
            schema=self._schema,
        )

    def fast_slice(self, offset=0, length=None):
        if offset < 0:
            raise IndexError("what are you doing? offset must be positive")
        elif offset >= self._offsets[-1] or (length is not None and length <= 0):
            return pa.Table.from_batches([], schema=self._schema)
        i = int(np.searchsorted(self._offsets, offset, side="right")) - 1
        if length is None or length + offset >= self._offsets[-1]:
            batches = self._batches[i:]
            batches[0] = batches[0].slice(offset - self._offsets[i])
        else:
            j = int(np.searchsorted(self._offsets, offset + length - 1, side="right")) - 1
            batches = self._batches[i:j + 1j]
            batches[-1] = batches[-1].slice(0, offset + length - self._offsets[j])
            batches[0] = batches[0].slice(offset - self._offsets[i])
        return pa.Table.from_batches(batches, schema=self._schema)


class Table(IndexPlugin):
    def __init__(self, table: pa.Table):
        super().__init__(table)
        self.table = table

    def __deepcopy__(self, memo: dict):
        memo[id(self.table)] = self.table
        memo[id(self._batches)] = self._batches
        return _deepcopy(self, memo)

    def validate(self, *args, **kwargs):
        return self.table.validate(*args, **kwargs)

    def equals(self, *args, **kwargs):
        args = tuple(arg.table if isinstance(arg, Table) else arg for arg in args)
        kwargs = {k: v.table if isinstance(v, Table) else v for k, v in kwargs.items()}
        return self.table.equals(*args, **kwargs)

    def to_batches(self, *args, **kwargs):
        return self.table.to_batches(*args, **kwargs)

    def to_pydict(self, *args, **kwargs):
        return self.table.to_pydict(*args, **kwargs)

    def to_pylist(self, *args, **kwargs):
        return self.table.to_pylist(*args, **kwargs)

    def to_pandas(self, *args, **kwargs):
        return self.table.to_pandas(*args, **kwargs)

    def to_string(self, *args, **kwargs):
        return self.table.to_string(*args, **kwargs)

    def to_reader(self, max_chunk_size: Optional[int] = None):
        return self.table.to_reader(max_chunksize=max_chunk_size)

    def field(self, *args, **kwargs):
        return self.table.field(*args, **kwargs)

    def column(self, *args, **kwargs):
        return self.table.column(*args, **kwargs)

    def iter_columns(self, *args, **kwargs):
        return self.table.itercolumns(*args, **kwargs)

    @property
    def columns(self):
        return self.table.columns

    @property
    def schema(self):
        return self.table.schema

    @property
    def num_columns(self):
        return self.table.num_columns

    @property
    def num_rows(self):
        return self.table.num_rows

    @property
    def shape(self):
        return self.table.shape

    @property
    def nbytes(self):
        return self.table.nbytes

    @property
    def column_names(self):
        return self.table.column_names

    def __eq__(self, other):
        return self.equals(other)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        return self.table[key]

    def __repr__(self):
        return self.table.__repr__().replace("pyarrow.Table", self.__class__.__name__)

    def __str__(self):
        return self.table.__str__().replace("pyarrow.Table", self.__class__.__name__)

    def slice(self, *args, **kwargs):
        raise NotImplementedError()

    def filter(self, *args, **kwargs):
        raise NotImplementedError()

    def flatten(self, *args, **kwargs):
        raise NotImplementedError()

    def combine_chunks(self, *args, **kwargs):
        raise NotImplementedError()

    def cast(self, *args, **kwargs):
        raise NotImplementedError()

    def replace_schema_metadata(self, *args, **kwargs):
        raise NotImplementedError()

    def add_column(self, *args, **kwargs):
        raise NotImplementedError()

    def append_column(self, *args, **kwargs):
        raise NotImplementedError()

    def remove_column(self, *args, **kwargs):
        raise NotImplementedError()

    def drop(self, *args, **kwargs):
        raise NotImplementedError()

    def select(self, *args, **kwargs):
        raise NotImplementedError()


class BlockTable(Table):
    """
    it is useful, don't remove
    """
    pass


class MemoryTable(BlockTable):
    @classmethod
    def from_storage(cls, filename: str):
        table = _memory_table_from_file(filename)
        return cls(table)

    @classmethod
    def from_buffer(cls, buffer: pa.Buffer):
        table = _memory_table_from_buffer(buffer)
        return cls(table)

    @classmethod
    def from_pandas(cls, *args, **kwargs):
        return cls(pa.Table.from_pandas(*args, **kwargs))

    @classmethod
    def from_arrays(cls, *args, **kwargs):
        return cls(pa.Table.from_arrays(*args, **kwargs))

    @classmethod
    def from_pydict(cls, *args, **kwargs):
        return cls(pa.Table.from_pydict(*args, **kwargs))

    @classmethod
    def from_pylist(cls, mapping, *args, **kwargs):
        return cls(pa.Table.from_pylist(mapping, *args, **kwargs))

    @classmethod
    def from_batches(cls, *args, **kwargs):
        return cls(pa.Table.from_batches(*args, **kwargs))

    def slice(self, offset=0, length=None):
        return MemoryTable(self.fast_slice(offset, length))

    def filter(self, *args, **kwargs):
        return MemoryTable(self.table.filter(*args, **kwargs))

    def flatten(self):
        """
        better than table.flatten
        Schema such as Image and Video wouldn't be flattened
        """
        return MemoryTable(table_flatten(self.table))

    def combine_chunks(self, *args, **kwargs):
        return MemoryTable(self.table.combine_chunks(*args, **kwargs))

    def cast(self, pa_schema: pa.Schema):
        return MemoryTable(table_cast(self.table, schema=pa_schema))

    def replace_schema_metadata(self, *args, **kwargs):
        return MemoryTable(self.table.replace_schema_metadata(*args, **kwargs))

    def add_column(self, *args, **kwargs):
        return MemoryTable(self.table.add_column(*args, **kwargs))

    def append_column(self, *args, **kwargs):
        return MemoryTable(self.table.append_column(*args, **kwargs))

    def remove_column(self, *args, **kwargs):
        return MemoryTable(self.table.remove_column(*args, **kwargs))

    def set_column(self, *args, **kwargs):
        return MemoryTable(self.table.set_column(*args, **kwargs))

    def rename_columns(self, *args, **kwargs):
        return MemoryTable(self.table.rename_columns(*args, **kwargs))

    def drop_column(self, *args, **kwargs):
        return MemoryTable(self.table.drop_column(*args, **kwargs))

    def select(self, *args, **kwargs):
        return MemoryTable(self.table.select(*args, **kwargs))


MemoryMappedTable_Replay = tuple[str, tuple, dict]


class MemoryMappedTable(BlockTable):
    def __init__(self, table: pa.Table, path: str, replays: Optional[list[MemoryMappedTable_Replay]] = None):
        super().__init__(table)
        self.path = os.path.abspath(path)
        self.replays: list[MemoryMappedTable_Replay] = replays if replays is not None else []

    @classmethod
    def from_file(cls, file_name: str, replays=None):
        table = _memory_mapped_arrow_table_from_file(file_name)
        table = cls._apply_replays(table, replays)
        return cls(table, file_name, replays)

    def __getstate__(self):
        return {"path": self.path, "replays": self.replays}

    def __setstate__(self, state):
        path = state["path"]
        replays = state["replays"]
        table = _memory_mapped_arrow_table_from_file(path)
        table = self._apply_replays(table, replays)
        MemoryMappedTable.__init__(self, table, path, replays)

    def _append_replay(self, replay: MemoryMappedTable_Replay):
        replays = copy.deepcopy(self.replays)
        replays.append(replay)
        return replays

    def slice(self, offset=0, length=None):
        replay = ("slice", (offset, length), {})
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.fast_slice(offset, length), self.path, replays)

    def filter(self, *args, **kwargs):
        replay = ("filter", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.filter(*args, **kwargs), self.path, replays)

    def flatten(self, *args, **kwargs):
        """
        better than table.flatten
        Schema such as Image and Video wouldn't be flattened
        """
        replay = ("flatten", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(table_flatten(self.table), self.path, replays)

    def combine_chunks(self, *args, **kwargs):
        replay = ("combine_chunks", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.combine_chunks(*args, **kwargs), self.path, replays)

    def cast(self, pa_schema: pa.Schema):
        replay = ("cast", (), {})
        replays = self._append_replay(replay)
        return MemoryMappedTable(table_cast(self.table, schema=pa_schema), self.path, replays)

    def replace_schema_metadata(self, *args, **kwargs):
        replay = ("replace_schema_metadata", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.replace_schema_metadata(*args, **kwargs), self.path, replays)

    def append_column(self, *args, **kwargs):
        replay = ("append_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.append_column(*args, **kwargs), self.path, replays)

    def remove_column(self, *args, **kwargs):
        replay = ("remove_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.remove_column(*args, **kwargs), self.path, replays)

    def set_column(self, *args, **kwargs):
        replay = ("set_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.set_column(*args, **kwargs), self.path, replays)

    def rename_columns(self, *args, **kwargs):
        replay = ("rename_columns", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.rename_columns(*args, **kwargs), self.path, replays)

    def drop_column(self, *args, **kwargs):
        replay = ("drop_column", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.drop_column(*args, **kwargs), self.path, replays)

    def select(self, *args, **kwargs):
        replay = ("select", copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.select(*args, **kwargs), self.path, replays)

    @staticmethod
    def _apply_replays(table: pa.Table, replays: Optional[list[MemoryMappedTable_Replay]]):
        if replays is not None:
            for name, args, kwargs in replays:
                if name == "cast":
                    table = table_cast(table, *args, **kwargs)
                elif name == "flatten":
                    table = table_flatten(table)
                else:
                    table = getattr(table, name)(*args, **kwargs)
        return table
