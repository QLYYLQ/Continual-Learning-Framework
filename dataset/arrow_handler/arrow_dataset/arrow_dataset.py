import json
from typing import Optional

import pyarrow as pa
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.dataset_plugin import DatasetInfoPlugin, DatasetInfo, \
    NamedSplit
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.fingerprint import generate_fingerprint
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.index_enhancement.dataset_index_plugin import \
    IndexablePlugin
from CLTrainingFramework.dataset.arrow_handler.arrow_reader import ArrowReader
from CLTrainingFramework.dataset.arrow_handler.arrow_table import Table, MemoryTable
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.utils import _check_table, register_dataset_for_no_cache, \
    _check_column_names, update_metadata_with_schema
from CLTrainingFramework.dataset.schema import Schema


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

    @property
    def schema(self) -> Schema:
        schema = super().schema
        if schema is None:
            raise ValueError("For dataset, schema can't be None")
        return schema

    @property
    def data(self) -> Table:
        return self._data

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
    )->"Dataset":
        table = MemoryTable.from_buffer(buffer)
        if indices_buffer is not None:
            indices_table = MemoryTable.from_buffer(indices_buffer)
        else:
            indices_table = None
        return cls(table=table, info=info, split=split, indices_table=indices_table)
    @classmethod
    def from_dict(cls,mapping:dict,schema:Optional[Schema]=None,info:Optional[DatasetInfo]=None,splt:Optional[NamedSplit]=None) -> "Dataset":
        if info is not None and schema is not None and info.schema != schema:
            raise ValueError("Schema and info.schema can't be different")
