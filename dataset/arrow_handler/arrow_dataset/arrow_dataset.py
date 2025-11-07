import copy
import json
import os.path
from os import PathLike
from pathlib import Path
from typing import Optional, Union, Callable, TypeVar, overload, Iterable

import fsspec
import pyarrow as pa
from fsspec import url_to_fs

import CLTrainingFramework.dataset.utils.dataset_config as dataset_config
from CLTrainingFramework.dataset.arrow_handler.arrow_table.table import table_cache_file_list
from CLTrainingFramework.dataset.arrow_handler.arrow_table.utils import table_iter, map_function_to_table
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.dataset_plugin import DatasetInfoPlugin, DatasetInfo, \
    NamedSplit
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.fingerprint import generate_fingerprint
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.index_enhancement.dataset_index_plugin import \
    IndexablePlugin
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.splits import Split
from CLTrainingFramework.dataset.arrow_handler.arrow_reader import ArrowReader
from CLTrainingFramework.dataset.arrow_handler.arrow_table import Table, MemoryTable
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.utils import _check_table, register_dataset_for_no_cache, \
    _check_column_names, update_metadata_with_schema
from CLTrainingFramework.dataset.arrow_handler.arrow_table.utils import cast_pa_array_using_schema
from CLTrainingFramework.dataset.arrow_handler.arrow_writer import OptimizedTypedSequence
from CLTrainingFramework.dataset.schema import Schema, pyarrow_to_schema, Video, Image, SchemaType
from CLTrainingFramework.dataset.schema.Schema import require_loading
from CLTrainingFramework.dataset.utils.fingerprint import fingerprint_transform
from CLTrainingFramework.dataset.formatting import get_formatter, format_table, query_table
from CLTrainingFramework.dataset.utils.py_utils_mine import convert_file_size_to_int

_T = TypeVar("_T")


class Dataset(DatasetInfoPlugin, IndexablePlugin):
    """A Dataset backed by an Arrow table."""

    def __init__(self, table: Table, info: DatasetInfo = None, split: Optional[NamedSplit] = None,
                 indices_table: Optional[Table] = None, fingerprint: Optional[str] = None):
        info = info.copy() if info is not None else DatasetInfo
        DatasetInfoPlugin.__init__(self, info, split)
        IndexablePlugin.__init__(self)
        self._data: Table = _check_table(table)
        self._indices: Optional[Table] = _check_table(indices_table) if indices_table is not None else None
        register_dataset_for_no_cache(self)
        self._format_type: Optional[str] = None
        self._format_kwargs: dict = {}
        self._format_columns: Optional[list] = None
        self._output_all_columns: bool = False
        self._fingerprint: str = fingerprint
        # metadata
        if self._data.schema.metadata is not None and b"CLTrainingFramework" in self._data.schema.metadata:
            metadata = json.load(self._data.schema.metadata[b"CLTrainingFramework"].decode())
            if (
                    "fingerprint" in metadata and self._fingerprint is None
            ):
                self._fingerprint: str = metadata["fingerprint"]

        # make sure schema is same
        inferred_schema = Schema.from_arrow_schema(table.schema)
        if self.info.schema is None:
            self.info.schema = inferred_schema
        else:
            try:
                self.info.schema = self.info.schema.reorder_fields_as(inferred_schema)
            except ValueError as e:
                raise ValueError(f"{e}\n好哥哥，info中记载的schema和传入数据的schema对不上")

        if self.data.schema != self.info.schema.to_arrow_schema():
            self._data = self.data.cast(self.info.schema.to_arrow_schema())

        if self._fingerprint is None:
            self._fingerprint = generate_fingerprint(self)

        if self._info.schema is None:
            raise ValueError("Schema can't be None in a Dataset object")
        if self._fingerprint is None:
            raise ValueError("Fingerprint can't be None in a Dataset object")
        if self.info.schema.type == inferred_schema.type:
            raise ValueError(
                f"External Schema info don't match, Got:\n{inferred_schema}\n for External Schema, but get \n{self.schema}\n for dataset")
        if self._indices is not None:
            if not pa.types.is_unsigned_integer(self._indices.column(0).type):
                raise ValueError(f"indices must be an Arrow table of unsigned integers, Got:\n{self._indices}")
        _check_column_names(self._data.column_names)
        self._data = update_metadata_with_schema(self._data, self._info.schema)

    def __setstate__(self, state):
        self.__dict__.update(state)
        register_dataset_for_no_cache(self)
        return self

    def __del__(self):
        if hasattr(self, "_data"):
            del self._data
        if hasattr(self, "_indices"):
            del self._indices

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def __len__(self):
        return self.num_rows

    def __iter__(self):
        if self._indices is None:
            format_kwargs = self._format_kwargs if self._format_kwargs is not None else {}
            formatter = get_formatter(self._format_type, schema=self._info.schema, **format_kwargs)
            batch_size = dataset_config.ARROW_READER_BATCH_SIZE_IN_DATASET_ITER
            for pa_sub_table in table_iter(self.data, batch_size=batch_size):
                for i in range(pa_sub_table.num_rows):
                    pa_sub_table_ex = pa_sub_table.slice(i, 1)
                    formatted_output = format_table(pa_sub_table_ex, 0, formatter=formatter,
                                                    format_columns=self._format_columns,
                                                    output_all_columns=self._output_all_columns)
        else:
            for i in range(self.num_rows):
                yield self._getitem(i)

    @overload
    def __getitem__(self, i: Union[int, slice, Iterable[int]]) -> dict:
        ...

    @overload
    def __getitem__(self, i: str) -> list:
        ...

    def __getitem__(self, i):
        return self._getitem(i)

    def __getitems__(self, key: list):
        batch = self.__getitem__(key)
        n_example = len(batch[next(iter(batch))])
        return [{k: v[i] for k, v in batch.items()} for i in range(n_example)]

    def _estimate_dataset_size(self) -> int:
        dataset_bytes = self.data.nbytes
        loading_columns = [
            k for k, v in self._info.schema.items()
            if require_loading(v, ignore_sample_from_load_attribute=True)
        ]
        if loading_columns:
            extra_bytes = 0
            table = self.with_format("arrow")[:1000]

            def embed_schema_helper(array, schema: SchemaType):
                nonlocal extra_bytes
                if isinstance(schema, (Video, Image)):
                    for x in array.to_pylist():
                        if x is not None and x["bytes"] is not None and x["path"] is not None:
                            size = os.path.getsize(x["path"])
                            extra_bytes += size
                    extra_bytes -= array.field("path").nbytes

            map_function_to_table(table, embed_schema_helper)
            extra_bytes = extra_bytes * len(self.data) / len(table)
            dataset_bytes = dataset_bytes + extra_bytes
        if self._indices is not None:
            dataset_bytes = dataset_bytes * len(self._indices) / len(self.data)
        return dataset_bytes

    def _getitem(self, i: Union[int, slice, str, list[_T], tuple[_T, ...]], **kwargs) -> Union[dict, list]:
        format_type = kwargs.get("format_type") if "format_type" in kwargs else self._format_type
        format_columns = kwargs.get("format_columns") if "format_columns" in kwargs else self._format_columns
        output_all_columns = (
            kwargs.get("output_all_columns") if "output_all_columns" in kwargs else self._output_all_columns)
        format_kwargs = self._format_kwargs if "format_kwargs" in kwargs else self._format_kwargs
        formatter = get_formatter(format_type, schema=self._info.schema, **format_kwargs)
        pa_sub_table = query_table(self._data, i, indices=self._indices)
        formatted_output = format_table(pa_sub_table, i, formatter, format_columns, output_all_columns)
        return formatted_output

    @property
    def schema(self) -> Schema:
        schema = super().schema
        if schema is None:
            raise ValueError("For dataset, schema can't be None")
        return schema

    @property
    def data(self) -> Table:
        return self._data

    @property
    def num_rows(self):
        if self._indices is not None:
            return self._indices.num_rows
        return self._data.num_rows

    @property
    def cache_files(self) -> list[dict]:
        cache_files = table_cache_file_list(self.data)
        if self._indices is not None:
            cache_files += table_cache_file_list(self._indices)
        return [{"filename": a} for a in cache_files]

    @classmethod
    def from_file(
            cls,
            filename: str,
            info: Optional[DatasetInfo] = None,
            split: Optional[NamedSplit] = None,
            indices_filename: Optional[str] = None,
            in_memory: bool = False,
    ) -> "Dataset":
        table = ArrowReader.read_table(filename, in_memory=in_memory)
        if indices_filename is not None:
            indices_pa_tabe = ArrowReader.read_table(indices_filename, in_memory=in_memory)
        else:
            indices_pa_tabe = None
        return cls(
            table=table,
            info=info,
            split=split,
            indices_table=indices_pa_tabe,
        )

    @classmethod
    def from_buffer(
            cls, buffer: pa.Buffer,
            info: Optional[DatasetInfo] = None,
            split: Optional[NamedSplit] = None,
            indices_buffer: Optional[pa.Buffer] = None,
    ) -> "Dataset":
        table = MemoryTable.from_buffer(buffer)
        if indices_buffer is not None:
            indices_table = MemoryTable.from_buffer(indices_buffer)
        else:
            indices_table = None
        return cls(table=table, info=info, split=split, indices_table=indices_table)

    @classmethod
    def from_dict(cls, mapping: dict, schema: Optional[Schema] = None, info: Optional[DatasetInfo] = None,
                  split: Optional[NamedSplit] = None) -> "Dataset":
        if info is not None and schema is not None and info.schema != schema:
            raise ValueError("Schema and info.schema can't be different")
        schema = schema if schema is not None else info.schema if info is not None else None
        arrow_mapping = {}
        for c, data in mapping.items():
            if isinstance(data, (pa.Array, pa.ChunkedArray)):
                data = cast_pa_array_using_schema(data, schema[c]) if schema is not None else data
            else:
                data = OptimizedTypedSequence(
                    schema.sample_to_pa_cache(data, c) if schema is not None else data,
                    type=schema[c] if schema is not None else None,
                    col=c
                )
            arrow_mapping[c] = data
        pa_table = MemoryTable.from_pydict(mapping=arrow_mapping)
        if info is None:
            info = DatasetInfo()
        info.schema = schema
        if info.schema is None:
            info.schema = Schema(
                {
                    column: pyarrow_to_schema(data.type)
                    if isinstance(data, (pa.Array, pa.ChunkedArray))
                    else data.get_inferred_type()
                    for column, data in arrow_mapping.items()
                }
            )
        return cls(table=pa_table, info=info, split=split)

    @classmethod
    def from_list(cls, mapping: list[dict], schema: Optional[Schema] = None, info: Optional[DatasetInfo] = None,
                  splt: Optional[NamedSplit] = None) -> "Dataset":
        # restructure the list mapping into dict mapping
        mapping = {k: [i.get(k) for i in mapping] for k in mapping[0]} if mapping else {}
        return cls.from_dict(mapping, schema, info, splt)

    @staticmethod
    def from_generator(generator: Callable, schema: Optional[Schema] = None, cache_dir: str = None,
                       keep_in_memory: bool = False, gen_kwargs: Optional[dict] = None, num_proc: Optional[int] = None,
                       split: NamedSplit = Split.TRAIN) -> "Dataset":
        raise NotImplementedError

    @staticmethod
    def from_json(path: Union[PathLike, list[PathLike]], split: Optional[NamedSplit] = None,
                  schema: Optional[Schema] = None, cache_dir: Optional[str] = None, keep_in_memory: bool = False,
                  field: Optional[str] = None, num_proc: Optional[int] = None, **kwargs):
        raise NotImplementedError

    def save_to_disk(self, dataset_path: PathLike, max_shard_size: Optional[Union[str, int]] = None,
                     num_shards: Optional[int] = None, num_proc: int = 1, storage_options: Optional[dict] = None):
        """
        args:
            dataset_path:
            max_shard_size:
            num_shards:
            num_proc:
            storage_options:
                options for fsspec.core.url_to_fs()
        """
        if max_shard_size is not None and num_shards is not None:
            raise ValueError("这两个值给一个就行")
        if self.list_indexes():
            raise ValueError("remove all the indexes using drop_index before saving")
        if num_shards is None:
            dataset_bytes = self._estimate_dataset_size()
            max_shard_size = convert_file_size_to_int(max_shard_size or dataset_config.MAX_SHARD_SIZE)
            num_shards = int(dataset_bytes / max_shard_size) + 1
        num_shards = num_shards if num_shards is not None else num_proc
        fs: fsspec.AbstractFileSystem
        fs, _ = url_to_fs(dataset_path, **(storage_options or {}))
        if True:  # for remote file system, this is not necessary
            parent_cache_file_path = {Path(a["filename"]).resolve().parent for a in self.cache_files}
            if Path(dataset_path).expanduser().resolve() in parent_cache_file_path:
                raise PermissionError("哥，你好像对一个文件同时在做读和写的操作，有permission error或者segfault会弹出来的，我先帮你Error一下")

    def with_format(self, type: Optional[str] = None, columns: Optional[list] = None, output_all_columns: bool = False,
                    **kwargs) -> "Dataset":
        dataset = copy.deepcopy(self)
        dataset.set_format(type=type, columns=columns, output_all_columns=output_all_columns, **kwargs)
        return dataset

    @fingerprint_transform(inplace=True)
    def set_format(self, type, columns, output_all_columns, **kwargs):
        pass
