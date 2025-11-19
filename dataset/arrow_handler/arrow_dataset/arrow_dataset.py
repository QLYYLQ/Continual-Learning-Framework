import copy
import itertools
import json
import os.path
import shutil
import tempfile
from os import PathLike
from pathlib import Path
from typing import Optional, Union, Callable, TypeVar, overload, Iterable, Iterator, Literal

import fsspec
import numpy as np
import pyarrow as pa
from fsspec import url_to_fs
from multiprocess import Pool

import CLTrainingFramework.dataset.utils.dataset_config as dataset_config
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.dataset_plugin import DatasetInfoPlugin, DatasetInfo, \
    NamedSplit
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.fingerprint import generate_fingerprint
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.index_enhancement.dataset_index_plugin import \
    IndexablePlugin
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.splits import Split
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.utils import _check_table, register_dataset_for_no_cache, \
    _check_column_names, update_metadata_with_schema, _using_cache, get_temp_cache_dir
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.utils import transmit_format
from CLTrainingFramework.dataset.arrow_handler.arrow_reader import ArrowReader
from CLTrainingFramework.dataset.arrow_handler.arrow_table import Table, MemoryTable, MemoryMappedTable
from CLTrainingFramework.dataset.arrow_handler.arrow_table.table import table_cache_file_list
from CLTrainingFramework.dataset.arrow_handler.arrow_table.utils import cast_pa_array_using_schema
from CLTrainingFramework.dataset.arrow_handler.arrow_table.utils import table_iter, map_function_to_table
from CLTrainingFramework.dataset.arrow_handler.arrow_writer import OptimizedTypedSequence, ArrowWriter
from CLTrainingFramework.dataset.formatting import get_formatter, format_table, query_table, is_range_contiguous, \
    get_format_type_from_alias
from CLTrainingFramework.dataset.schema import Schema, pyarrow_to_schema, Video, Image, SchemaType
from CLTrainingFramework.dataset.schema.Schema import require_loading
from CLTrainingFramework.dataset.utils.fingerprint import fingerprint_transform, generate_random_fingerprint, \
    format_transform_for_fingerprint, format_kwargs_for_fingerprint, update_fingerprint, validate_fingerprint
from CLTrainingFramework.dataset.utils.py_utils_mine import convert_file_size_to_int, as_dict, \
    parallel_flatmap_unordered
from CLTrainingFramework.utils.global_tqdm import tqdm
from CLTrainingFramework.utils.logging import get_logger

logger = get_logger(__name__)
_T = TypeVar("_T")


def _check_valid_indices_value(index, size):
    if (index < 0 and index + size < 0) or (index >= size):
        raise IndexError(f"Index {index} out of range for dataset of size {size}")


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

    @transmit_format
    @fingerprint_transform(inplace=False)
    def _select_contiguous(self, start: int, length: int, new_fingerprint: Optional[str] = None) -> "Dataset":
        if len(self.list_indexes()) > 0:
            raise RuntimeError(
                "first run drop_index to remove index and then re-add it"
            )
        if len(self) == 0:
            return self
        _check_valid_indices_value(start, len(self))
        _check_valid_indices_value(start + length - 1, len(self))
        if self._indices is not None or length == 0:
            return Dataset(
                self.data.slice(start, length),
                info=self.info.copy(),
                split=self.split,
                fingerprint=new_fingerprint
            )
        else:
            return Dataset(
                self.data,
                info=self.info.copy(),
                split=self.split,
                indices_table=self._indices.slice(start, length),
                fingerprint=new_fingerprint
            )

    @transmit_format
    @fingerprint_transform(inplace=False)
    def _select_with_indices_mapping(self, indices: Iterable, keep_in_memory: bool = False,
                                     indices_cache_file_name: Optional[str] = None,
                                     writer_batch_size: Optional[int] = 1000,
                                     new_fingerprint: Optional[str] = None) -> "Dataset":
        """Create a new dataset with rows selected following the list/array of indices.
                The new dataset is made by creating a new indices mapping on top of the main arrow table.

                Args:
                    indices (sequence, iterable, range, ndarray or Series): List or 1D-array of integer indices for indexing.
                    keep_in_memory (`bool`, default `False`): Keep the indices mapping in memory instead of writing it to a cache file.
                    indices_cache_file_name (`str`, optional, default `None`): Provide the name of a path for the cache file. It is used to store the
                        indices mapping instead of the automatically generated cache file name.
                    writer_batch_size (`int`, default `1000`): Number of rows per write operation for the cache file writer.
                        This value is a good trade-off between memory usage during the processing, and processing speed.
                        Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `.map()`.
                    new_fingerprint (`str`, optional, default `None`): the new fingerprint of the dataset after transform.
                        If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
                """
        if keep_in_memory and indices_cache_file_name is not None:
            raise ValueError("Please use either `keep_in_memory` or `indices_cache_file_name` but not both.")

        if len(self.list_indexes()) > 0:
            raise RuntimeError(
                "Using `.select` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it."
            )

        # If the array is empty we do nothing
        if len(self) == 0:
            return self

        # Prepare the writer for our indices arrow table
        if keep_in_memory or indices_cache_file_name is None:
            buf_writer = pa.BufferOutputStream()
            tmp_file = None
            writer = ArrowWriter(
                stream=buf_writer, writer_batch_size=writer_batch_size, fingerprint=new_fingerprint, unit="indices"
            )
        else:
            buf_writer = None
            logger.info(f"Caching indices mapping at {indices_cache_file_name}")
            cache_dir = os.path.dirname(indices_cache_file_name)
            os.makedirs(cache_dir, exist_ok=True)
            tmp_file = tempfile.NamedTemporaryFile("wb", dir=cache_dir, delete=False)
            writer = ArrowWriter(
                path=tmp_file.name, writer_batch_size=writer_batch_size, fingerprint=new_fingerprint, unit="indices"
            )

        indices = indices if isinstance(indices, list) else list(indices)

        size = len(self)
        if indices:
            _check_valid_indices_value(int(max(indices)), size=size)
            _check_valid_indices_value(int(min(indices)), size=size)
        else:
            return self._select_contiguous(0, 0, new_fingerprint=new_fingerprint)

        indices_array = pa.array(indices, type=pa.uint64())
        # Check if we need to convert indices
        if self._indices is not None:
            indices_array = self._indices.column(0).take(indices_array)

        indices_table = pa.Table.from_arrays([indices_array], names=["indices"])

        with writer:
            try:
                writer.write_table(indices_table)
                writer.finalize()  # close_stream=bool(buf_writer is None))  We only close if we are writing in a file
            except (Exception, KeyboardInterrupt):
                if tmp_file is not None:
                    tmp_file.close()
                    if os.path.exists(tmp_file.name):
                        os.remove(tmp_file.name)
                raise

        if tmp_file is not None:
            tmp_file.close()
            shutil.move(tmp_file.name, indices_cache_file_name)
            umask = os.umask(0o666)
            os.umask(umask)
            os.chmod(indices_cache_file_name, 0o666 & ~umask)

        # Return new Dataset object
        if buf_writer is None:
            return self._new_dataset_with_indices(
                indices_cache_file_name=indices_cache_file_name, fingerprint=new_fingerprint
            )
        else:
            return self._new_dataset_with_indices(indices_buffer=buf_writer.getvalue(), fingerprint=new_fingerprint)

    def _new_dataset_with_indices(self, indices_cache_file_name: Optional[str] = None,
                                  indices_buffer: Optional[pa.Buffer] = None, fingerprint: Optional[str] = None):
        if indices_cache_file_name is None and indices_buffer is None:
            raise ValueError(
                "Either `indices_cache_file_name` or `indices_buffer` must be provided."
            )
        if fingerprint is None:
            raise ValueError(
                "fingerprint is needed"
            )
        if indices_cache_file_name is not None:
            indices_table = MemoryMappedTable.from_file(indices_cache_file_name)
        else:
            indices_table = MemoryTable.from_buffer(indices_buffer)
        return Dataset(
            self.data,
            info=self.info.copy(),
            split=self.split,
            indices_table=indices_table,
            fingerprint=fingerprint,
        )

    def _get_cache_file(self, fingerprint: str) -> str:
        if _using_cache():
            cache_file_name = "arrow-" + fingerprint + ".arrow"
            _cache_dir = os.path.dirname(self.cache_files[0]["filename"])
        else:
            cache_file_name = "arrow-" + generate_random_fingerprint() + ".arrow"
            _cache_dir = get_temp_cache_dir()
        full_path = os.path.join(_cache_dir, cache_file_name)
        return full_path

    @staticmethod
    def _map_single(
            shard: "Dataset",
            function: Optional[Callable] = None,
            with_indices: bool = False,
            with_rank: bool = False,
            input_columns: Optional[list[str]] = None,
            batched: bool = False,
            batch_size: Optional[int] = 1000,
            drop_last_batch: bool = False,
            remove_columns: Optional[list[str]] = None,
            keep_in_memory: bool = False,
            cache_file_name: Optional[str] = None,
            writer_batch_size: Optional[int] = 1000,
            schema: Optional[Schema] = None,
            disable_nullable: bool = False,
            fn_kwargs: Optional[dict] = None,
            new_fingerprint: Optional[str] = None,
            rank: Optional[int] = None,
            offset: int = 0,
            try_original_type: Optional[bool] = True,
    ) -> Iterable[tuple[int, bool, Union[int, "Dataset"]]]:
        raise NotImplementedError

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

    @property
    def column_names(self) -> list[str]:
        return self._data.column_names

    @property
    def format(self):
        return {
            "type": self._format_type,
            "format_kwargs": self._format_kwargs,
            "columns": self.column_names if self._format_columns is None else self._format_columns,
            "output_all_columns": self._output_all_columns,
        }

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

    def save_to_disk(self, dataset_path: Union[str, PathLike, bytes], max_shard_size: Optional[Union[str, int]] = None,
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
                raise PermissionError(
                    "哥，你好像对一个文件同时在做读和写的操作，有permission error或者segfault会弹出来的，我先帮你Error一下")
        fs.makedirs(dataset_path, exist_ok=True)
        state = {
            i: self.__dict__[i]
            for i in dataset_config.DATASET_CACHE_ATTR
        }
        state["_split"] = str(self.split) if self.split is not None else self.split
        state["_data_files"] = [
            {
                "filename": f"data-{i:05d}-of-{num_shards:05d}.arrow" for i in range(num_shards)
            }
        ]
        for k, v in state["_format_kwargs"].items():
            try:
                json.dumps(v)
            except TypeError as e:
                raise TypeError(
                    f"str(e)\n---From Framework\nThe format kwargs must be JSON serializable, the value:{v}, with key:{k} can't be serializable"
                )
        dataset_info = as_dict(self._info)
        kwargs_per_job = (
            {
                "job_id": i,
                "shard": self.shard()
            }
            for i in range(num_shards)
        )

    def shard(
            self,
            num_shards: int,
            index: int,
            contiguous: bool = True,
            keep_in_memory: bool = False,
            indices_cache_file_name: Optional[str] = None,
            writer_batch_size: Optional[int] = 1000,
    ) -> "Dataset":
        """
        Return the `index`-nth shard from dataset split into `num_shards` pieces.

        Note: n should be less or equal to the number of elements in the dataset `len(dataset)`.

        On the other hand, `dataset.shard(n, i, contiguous=False)` contains all elements of the dataset whose index mod `n = i`.

        Be sure to shard before using any randomizing operator (such as `shuffle`).
        It is best if the shard operator is used early in the dataset pipeline.

        Args:
            num_shards (`int`):
                How many shards to split the dataset into.
            index (`int`):
                Which shard to select and return.
            contiguous: (`bool`, defaults to `True`):
                Whether to select contiguous blocks of indices for shards.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            indices_cache_file_name (`str`, *optional*):
                Provide the name of a path for the cache file. It is used to store the
                indices of each shard instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                This only concerns the indices mapping.
                Number of indices per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
        """
        if not 0 <= index < num_shards:
            raise ValueError("index should be in [0, num_shards-1]")
        if contiguous:
            div = len(self) // num_shards
            mod = len(self) % num_shards
            start = div * index + min(index, mod)
            end = start + div + (1 if index < mod else 0)
            indices = range(start, end)
        else:
            indices = np.arange(index, len(self), num_shards)

        return self.select(
            indices=indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
        )

    def with_format(self, type: Optional[str] = None, columns: Optional[list] = None, output_all_columns: bool = False,
                    **kwargs) -> "Dataset":
        dataset = copy.deepcopy(self)
        dataset.set_format(type=type, columns=columns, output_all_columns=output_all_columns, **kwargs)
        return dataset

    def with_transform(self, transform: Optional[Callable], columns: Optional[list] = None,
                       output_all_columns: bool = False) -> "Dataset":
        dataset = copy.deepcopy(self)
        dataset.set_transform(transform=transform, columns=columns, output_all_columns=output_all_columns)
        return dataset

    @fingerprint_transform(inplace=True)
    def set_format(self, type: Optional[str] = None, columns: Optional[list] = None, output_all_columns: bool = False,
                   **kwargs):
        kwargs.update(kwargs.pop("format_kwargs", {}))
        type = get_format_type_from_alias(type)
        # 这里不需要返回值，只是做过检查，type和其他能否匹配
        get_formatter(type, schema=self._info.schema, **kwargs)
        if isinstance(columns, str):
            columns = [columns]
        elif isinstance(columns, tuple):
            columns = list(columns)
        if columns is not None:
            missing_columns = set(columns) - set(self._data.column_names)
            if missing_columns:
                raise ValueError(
                    f"Columns {list(missing_columns)} not in the dataset, current columns is:{self._data.column_names}\nyour columns is:{columns}"
                )
            columns = columns.copy()  # 别搞点幺蛾子出来
        self._format_type = type
        self._format_kwargs = kwargs
        self._format_columns = columns
        self._output_all_columns = output_all_columns
        logger.debug(
            f"Set __getitem__ output type to \n{type}\n for columns \n{columns}\n"
        )

    def reset_format(self):
        self.set_format()

    def set_transform(self, transform: Optional[Callable], columns: Optional[list] = None,
                      output_all_columns: bool = False):
        self.set_format("custom", columns=columns, output_all_columns=output_all_columns, transform=transform)

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["indices_cache_file_name"])
    def select(self, indices: Iterable, keep_in_memory: bool = False, indices_cache_file_name: Optional[str] = None,
               writer_batch_size: Optional[int] = 1000, new_fingerprint: Optional[str] = None) -> "Dataset":
        if keep_in_memory and indices_cache_file_name is not None:
            raise ValueError("not both keep in memory or indices cache file name")
        if len(self.list_indexes()) > 0:
            raise RuntimeError(
                "using select on a dataset with attached indexes is not allowed, first run drop_index to remove index and then re-add it")
        if len(self) == 0:
            return self
        if isinstance(indices, (pa.Array, pa.ChunkedArray)):
            indices = indices.to_numpy().astype(np.int64)
        elif isinstance(indices, Iterator):
            indices = list(indices)
        elif isinstance(indices, range):
            if is_range_contiguous(indices) and indices.start >= 0:
                start, length = indices.start, indices.stop - indices.start
                return self._select_contiguous(start, length, new_fingerprint=new_fingerprint)
        else:
            try:
                start = next(iter(indices))
            except StopIteration as e:
                return self._select_contiguous(0, 0, new_fingerprint=new_fingerprint)
            if start >= 0:
                counter_from_start = itertools.count(start=start)
                if all(i == j for i, j in zip(indices, counter_from_start)):
                    length = next(counter_from_start) - start
                    return self._select_contiguous(start, length, new_fingerprint)
        return self._select_with_indices_mapping(
            indices, keep_in_memory, indices_cache_file_name, writer_batch_size, new_fingerprint
        )

    @transmit_format
    def map(self, function: Optional[Callable] = None, with_indices: bool = False, with_rank: bool = False,
            input_columns: Optional[Union[str, list[str]]] = None, batched: bool = False,
            batch_size: Optional[int] = 1000, drop_last_batch: bool = False,
            remove_columns: Optional[Union[str, list[str]]] = None, keep_in_memory: bool = False,
            load_from_cache_file: Optional[bool] = None, cache_file_name: Optional[str] = None,
            writer_batch_size: Optional[int] = 1000, schema: Optional[Schema] = None,
            disable_nullable: bool = False, fn_kwargs: Optional[dict] = None, num_proc: Optional[int] = None,
            suffix_template: str = "_{rank:05d}_of_{num_proc:05d}", new_fingerprint: Optional[str] = None,
            desc: Optional[str] = None, try_original_type: Optional[bool] = True,
            ) -> "Dataset":
        if keep_in_memory and cache_file_name is not None:
            raise ValueError("not both keep in memory or indices cache file name")
        if num_proc is not None and num_proc <= 0:
            raise ValueError("num_proc must be greater than 0")
        if len(self) == 0:
            if self._indices is not None:
                self = Dataset(self.data.slice(0, 0),
                               info=self.info.copy(),
                               split=self.split, fingerprint=new_fingerprint)
            if remove_columns is not None:
                return self.remove_columns(remove_columns)
            else:
                return self
        if function is None:
            function = lambda x: x
        if isinstance(input_columns, str):
            input_columns = [input_columns]
        if input_columns is not None:
            missing_columns = set(input_columns) - set(self._data.column_names)
            if missing_columns:
                raise ValueError(
                    f"{missing_columns} not in the dataset.\ndataset has columns: {self._data.column_names}")
        load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else _using_cache()
        if fn_kwargs is None:
            fn_kwargs = {}
        if num_proc is not None and num_proc > len(self):
            num_proc = len(self)
            logger.warning(
                f"num_proc={num_proc} > len(self)={len(self)}, reset num_proc to the dataset size: {len(self)}")

        dataset_kwargs = {
            "shard": self,
            "function": function,
            "with_indices": with_indices,
            "with_rank": with_rank,
            "input_columns": input_columns,
            "batched": batched,
            "batch_size": batch_size,
            "drop_last_batch": drop_last_batch,
            "remove_columns": remove_columns,
            "keep_in_memory": keep_in_memory,
            "writer_batch_size": writer_batch_size,
            "schema": schema,
            "disable_nullable": disable_nullable,
            "fn_kwargs": fn_kwargs,
            "try_original_type": try_original_type,
        }

        if new_fingerprint is None:
            # we create a unique hash from the function,
            # current dataset file and the mapping args
            transform = format_transform_for_fingerprint(Dataset._map_single)
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(Dataset._map_single, (), dataset_kwargs)
            kwargs_for_fingerprint["fingerprint_name"] = "new_fingerprint"
            new_fingerprint = update_fingerprint(self._fingerprint, transform, kwargs_for_fingerprint)
        else:
            validate_fingerprint(new_fingerprint)
        dataset_kwargs["new_fingerprint"] = new_fingerprint

        if self.cache_files:
            if cache_file_name is None:
                cache_file_name = self._get_cache_file(new_fingerprint)
        dataset_kwargs["cache_file_name"] = cache_file_name

        def load_processed_shard_from_cache(shard_kwargs):
            """Load a processed shard from cache if it exists, otherwise throw an error."""
            shard = shard_kwargs["shard"]
            # Check if we've already cached this computation (indexed by a hash)
            if shard_kwargs["cache_file_name"] is not None:
                if os.path.exists(shard_kwargs["cache_file_name"]) and load_from_cache_file:
                    info = shard.info.copy()
                    info.schema = schema
                    return Dataset.from_file(shard_kwargs["cache_file_name"], info=info, split=shard.split)
            raise RuntimeError("Not such dataset")

        num_shards = num_proc if num_proc is not None else 1
        if batched and drop_last_batch:
            pbar_total = len(self) // num_shards // batch_size * num_shards * batch_size
        else:
            pbar_total = len(self)

        shards_done = 0
        if num_proc is None or num_proc == 1:
            transformed_dataset = None
            try:
                transformed_dataset = load_processed_shard_from_cache(dataset_kwargs)
                logger.info(f"Loading cached processed dataset at {dataset_kwargs['cache_file_name']}")
            except RuntimeError:
                pass
            if transformed_dataset is None:
                with tqdm(unit=" examples", total=pbar_total,desc=desc or "CLTrainingFramework.dataset.Dataset.map") as pbar:
                    for rank, done, content in Dataset._map_single(**dataset_kwargs):
                        if done:
                            shards_done += 1
                            logger.debug(f"Finished processing shard number {rank} of {num_shards}.")
                            transformed_dataset = content
                        else:
                            pbar.update(content)
            assert transformed_dataset is not None, "Failed to retrieve the result from map"
            # update fingerprint if the dataset changed
            if transformed_dataset._fingerprint != self._fingerprint:
                transformed_dataset._fingerprint = new_fingerprint
            return transformed_dataset
        else:

            def format_cache_file_name(
                    cache_file_name: Optional[str],
                    rank: Union[int, Literal["*"]],  # noqa: F722
            ) -> Optional[str]:
                if not cache_file_name:
                    return cache_file_name
                sep = cache_file_name.rindex(".")
                base_name, extension = cache_file_name[:sep], cache_file_name[sep:]
                if isinstance(rank, int):
                    cache_file_name = base_name + suffix_template.format(rank=rank, num_proc=num_proc) + extension
                    logger.info(f"Process #{rank} will write at {cache_file_name}")
                else:
                    cache_file_name = (
                            base_name
                            + suffix_template.replace("{rank:05d}", "{rank}").format(rank=rank, num_proc=num_proc)
                            + extension
                    )
                return cache_file_name

            def format_new_fingerprint(new_fingerprint: str, rank: int) -> str:
                new_fingerprint = new_fingerprint + suffix_template.format(rank=rank, num_proc=num_proc)
                validate_fingerprint(new_fingerprint)
                return new_fingerprint

            prev_env = copy.deepcopy(os.environ)
            # check if parallelism is off
            # from https://github.com/huggingface/tokenizers/blob/bb668bc439dc34389b71dbb8ce0c597f15707b53/tokenizers/src/utils/parallelism.rs#L22
            if prev_env.get("TOKENIZERS_PARALLELISM", "false").lower() not in (
                    "",
                    "off",
                    "false",
                    "f",
                    "no",
                    "n",
                    "0",
            ):
                logger.warning("Setting TOKENIZERS_PARALLELISM=false for forked processes.")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            shards = [
                self.shard(num_shards=num_proc, index=rank, contiguous=True, keep_in_memory=keep_in_memory)
                for rank in range(num_proc)
            ]
            kwargs_per_job = [
                {
                    **dataset_kwargs,
                    "shard": shards[rank],
                    "cache_file_name": format_cache_file_name(cache_file_name, rank),
                    "rank": rank,
                    "offset": sum(len(s) for s in shards[:rank]),
                    "new_fingerprint": format_new_fingerprint(new_fingerprint, rank),
                }
                for rank in range(num_shards)
            ]

            transformed_shards = [None] * num_shards
            for i in range(num_shards):
                try:
                    transformed_shards[i] = load_processed_shard_from_cache(kwargs_per_job[i])
                    kwargs_per_job[i] = None
                except RuntimeError :
                    pass

            kwargs_per_job = [kwargs for kwargs in kwargs_per_job if kwargs is not None]

            # We try to create a pool with as many workers as dataset not yet cached.
            if kwargs_per_job:
                if len(kwargs_per_job) < num_shards:
                    logger.info(
                        f"Reprocessing {len(kwargs_per_job)}/{num_shards} shards because some of them were missing from the cache."
                    )
                with Pool(len(kwargs_per_job)) as pool:
                    os.environ = prev_env
                    logger.info(f"Spawning {num_proc} processes")
                    with tqdm(
                            unit=" examples",
                            total=pbar_total,
                            desc=(desc or "CLTrainingFramework.dataset.Dataset.Map") + f" (num_proc={num_proc})",
                    ) as pbar:
                        for rank, done, content in parallel_flatmap_unordered(
                                pool, Dataset._map_single, iterable_kwargs=kwargs_per_job
                        ):
                            if done:
                                shards_done += 1
                                logger.debug(f"Finished processing shard number {rank} of {num_shards}.")
                                transformed_shards[rank] = content
                            else:
                                pbar.update(content)
                    pool.close()
                    pool.join()
                # Avoids PermissionError on Windows (the error: https://github.com/huggingface/datasets/actions/runs/4026734820/jobs/6921621805)
                for kwargs in kwargs_per_job:
                    del kwargs["shard"]
            else:
                logger.info(f"Loading cached processed dataset at {format_cache_file_name(cache_file_name, '*')}")
            if None in transformed_shards:
                raise ValueError(
                    f"Failed to retrieve results from map: result list {transformed_shards} still contains None - at "
                    "least one worker failed to return its results"
                )
            logger.info(f"Concatenating {num_proc} shards")
            result = _concatenate_map_style_datasets(transformed_shards)
            # update fingerprint if the dataset changed
            if any(
                    transformed_shard._fingerprint != shard._fingerprint
                    for transformed_shard, shard in zip(transformed_shards, shards)
            ):
                result._fingerprint = new_fingerprint
            else:
                result._fingerprint = self._fingerprint
            return result

    @transmit_format
    @fingerprint_transform(inplace=False)
    def remove_columns(self, column_names: Union[str, list[str]], new_fingerprint: Optional[str] = None) -> "Dataset":
        dataset = copy.deepcopy(self)
        if isinstance(column_names, str):
            column_names = [column_names]
        missing_column = set(column_names) - set(self._data.column_names)
        if missing_column:
            raise ValueError(
                f"{missing_column} not in the dataset.\ndataset has columns: {self._data.column_names}"
            )
        for column in column_names:
            del dataset._info.schema[column]
        dataset._data = dataset._data.drop(column_names)
        dataset._data = update_metadata_with_schema(dataset._data, dataset.schema)
        dataset._fingerprint = new_fingerprint
        return dataset

    def clean_cache(self) -> int:
        current_cache = [os.path.abspath(i["filename"]) for i in self.cache_files]
        if not current_cache:
            return 0
        cache_directory = os.path.dirname(current_cache[0])
        logger.info(f"Cache file in:{cache_directory}")
        files: list[str] = os.listdir(cache_directory)
        remove_count = 0
        for i in files:
            full_path = os.path.abspath(os.path.join(cache_directory, i))
            if i.startswith("cache-") and i.endswith(".arrow"):
                logger.info(f"Removing cache file:{full_path}")
                os.remove(full_path)
                remove_count += 1
        return remove_count
