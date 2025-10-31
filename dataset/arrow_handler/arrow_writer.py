# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""To write records into Parquet files."""

import json
import sys
from collections.abc import Iterable
from typing import Any, Optional, Union

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

# import config
import CLTrainingFramework.dataset.utils.dataset_config as dataset_config
import CLTrainingFramework.dataset.utils.package_config as package_config
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.dataset_info import DatasetInfo
from CLTrainingFramework.dataset.arrow_handler.arrow_table.utils import  cast_pa_array_using_schema, pa_table_cast
from CLTrainingFramework.dataset.arrow_handler.utils import DuplicatedKeysError, KeyHasher
from CLTrainingFramework.dataset.arrow_utils import _ArrayXDExtensionType, list_of_np_array_to_pyarrow_list_array, \
    numpy_to_pyarrow_list_array, to_pyarrow_list_array,for_storage,array_cast
from CLTrainingFramework.dataset.schema import Schema, Image, Value, Video, map_nested_schema, SchemaType, \
    pyarrow_to_schema, schema_to_pyarrow, prepare_for_pa_cache
from CLTrainingFramework.dataset.utils import logging
from CLTrainingFramework.dataset.utils.py_utils_mine import as_dict, first_non_null_non_empty_value
from CLTrainingFramework.utils.filesystem import is_remote_filesystem

logger = logging.get_logger(__name__)

type_ = type  # keep python's type function


def get_writer_batch_size(schema: Optional[Schema]) -> Optional[int]:
    """
    Get the writer_batch_size that defines the maximum row group size in the parquet files.
    The default in `datasets` is 1,000, but we lower it to 100 for image/audio datasets and 10 for videos.
    This allows to optimize random access to parquet file, since accessing 1 row requires
    to read its entire row group.

    This can be improved to get optimized size for querying/iterating
    but at least it matches the dataset viewer expectations on HF.

    Args:
        schema (`datasets.Features` or `None`):
            Dataset Features from `datasets`.
    Returns:
        writer_batch_size (`Optional[int]`):
            Writer batch size to pass to a dataset builder.
            If `None`, then it will use the `datasets` default.
    """
    if not schema:
        return None

    batch_size = np.inf

    def set_batch_size(feature: SchemaType) -> None:
        nonlocal batch_size
        if isinstance(feature, Image):
            batch_size = min(batch_size, dataset_config.PARQUET_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS)
        # elif isinstance(feature, Audio):
        #     batch_size = min(batch_size, dataset_config.PARQUET_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS)
        elif isinstance(feature, Video):
            batch_size = min(batch_size, dataset_config.PARQUET_ROW_GROUP_SIZE_FOR_VIDEO_DATASETS)
        elif isinstance(feature, Value) and feature.dtype == "binary":
            batch_size = min(batch_size, dataset_config.PARQUET_ROW_GROUP_SIZE_FOR_BINARY_DATASETS)

    map_nested_schema(schema, set_batch_size)

    return None if batch_size is np.inf else batch_size


class SchemaInferenceError(ValueError):
    pass


class TypedSequence:
    """
    This data container generalizes the typing when instantiating pyarrow arrays, tables or batches.

    More specifically it adds several features:
    - Support extension types like ``datasets.features.Array2DExtensionType``:
        By default pyarrow arrays don't return extension arrays. One has to call
        ``pa.ExtensionArray.from_storage(type, pa.array(data, type.storage_type))``
        in order to get an extension array.
    - Support for ``try_type`` parameter that can be used instead of ``type``:
        When an array is transformed, we like to keep the same type as before if possible.
        For example when calling :func:`datasets.Dataset.map`, we don't want to change the type
        of each column by default.
    - Better error message when a pyarrow array overflows.

    Example::

        from datasets.features import Array2D, Array2DExtensionType, Value
        from datasets.arrow_writer import TypedSequence
        import pyarrow as pa

        arr = pa.array(TypedSequence([1, 2, 3], type=Value("int32")))
        assert arr.type == pa.int32()

        arr = pa.array(TypedSequence([1, 2, 3], try_type=Value("int32")))
        assert arr.type == pa.int32()

        arr = pa.array(TypedSequence(["foo", "bar"], try_type=Value("int32")))
        assert arr.type == pa.string()

        arr = pa.array(TypedSequence([[[1, 2, 3]]], type=Array2D((1, 3), "int64")))
        assert arr.type == Array2DExtensionType((1, 3), "int64")

        table = pa.Table.from_pydict({
            "image": TypedSequence([[[1, 2, 3]]], type=Array2D((1, 3), "int64"))
        })
        assert table["image"].type == Array2DExtensionType((1, 3), "int64")

    """

    def __init__(
            self,
            data: Iterable,
            type: Optional[SchemaType] = None,
            try_type: Optional[SchemaType] = None,
            optimized_int_type: Optional[SchemaType] = None,
    ):
        # assert type is None or try_type is None,
        if type is not None and try_type is not None:
            raise ValueError("You cannot specify both type and try_type")
        # set attributes
        self.data = data
        self.type = type
        self.try_type = try_type  # is ignored if it doesn't match the data
        self.optimized_int_type = optimized_int_type
        # when trying a type (is ignored if data is not compatible)
        self.trying_type = self.try_type is not None
        self.trying_int_optimization = optimized_int_type is not None and type is None and try_type is None
        # used to get back the inferred type after __arrow_array__() is called once
        self._inferred_type = None

    def get_inferred_type(self) -> SchemaType:
        """Return the inferred feature type.
        This is done by converting the sequence to an Arrow array, and getting the corresponding
        feature type.

        Since building the Arrow array can be expensive, the value of the inferred type is cached
        as soon as pa.array is called on the typed sequence.

        Returns:
            SchemaType: inferred feature type of the sequence.
        """
        if self._inferred_type is None:
            # https://arrow.apache.org/docs/python/generated/pyarrow.array.html#pyarrow.array
            self._inferred_type = pyarrow_to_schema(pa.array(self).type)
        return self._inferred_type

    @staticmethod
    def _infer_custom_type_and_to_storage(data: Iterable) -> tuple[Iterable, Optional[SchemaType]]:
        """Implement type inference for custom objects like PIL.Image.Image -> Image type.

        This function is only used for custom python objects that can't be directly passed to build
        an Arrow array. In such cases is infer the feature type to use, and it encodes the data so
        that they can be passed to an Arrow array.

        Args:
            data (Iterable): array of data to infer the type, e.g. a list of PIL images.

        Returns:
            Tuple[Iterable, Optional[SchemaType]]: a tuple with:
                - the (possibly encoded) array, if the inferred feature type requires encoding
                - the inferred feature type if the array is made of supported custom objects like
                    PIL images, else None.
        """
        if package_config.PIL_AVAILABLE and "PIL" in sys.modules:
            import PIL.Image

            non_null_idx, non_null_value = first_non_null_non_empty_value(data)
            if isinstance(non_null_value, PIL.Image.Image):
                return [Image().sample_to_pa_cache(value) if value is not None else None for value in data], Image()
            if isinstance(non_null_value, list) and isinstance(non_null_value[0], PIL.Image.Image):
                return [[Image().sample_to_pa_cache(x) for x in value] if value is not None else None for value in
                        data], [
                    Image()
                ]
        # if config.PDFPLUMBER_AVAILABLE and "pdfplumber" in sys.modules:
        #     import pdfplumber
        #
        #     non_null_idx, non_null_value = first_non_null_non_empty_value(data)
        #     if isinstance(non_null_value, pdfplumber.pdf.PDF):
        #         return [Pdf().encode_example(value) if value is not None else None for value in data], Pdf()
        #     if isinstance(non_null_value, list) and isinstance(non_null_value[0], pdfplumber.pdf.PDF):
        #         return [[Pdf().encode_example(x) for x in value] if value is not None else None for value in data], [
        #             Pdf()
        #         ]
        return data, None

    def __arrow_array__(self, type: Optional[pa.DataType] = None):
        """
        This function is called when calling pa.array(typed_sequence)

        看不懂思密达，直接抄了
        """

        if type is not None:
            raise ValueError("TypedSequence is supposed to be used with pa.array(typed_sequence, type=None)")
        del type  # make sure we don't use it
        data = self.data
        # automatic type inference for custom objects
        if self.type is None and self.try_type is None:
            data, self._inferred_type = self._infer_custom_type_and_to_storage(data)
        if self._inferred_type is None:
            type = self.try_type if self.trying_type else self.type
        else:
            type = self._inferred_type
        pa_type = schema_to_pyarrow(type) if type is not None else None
        optimized_int_pa_type = (
            schema_to_pyarrow(self.optimized_int_type) if self.optimized_int_type is not None else None
        )
        trying_cast_to_python_objects = False
        try:
            # custom pyarrow types
            if isinstance(pa_type, _ArrayXDExtensionType):
                storage = to_pyarrow_list_array(data, pa_type)
                return pa.ExtensionArray.from_storage(pa_type, storage)

            # efficient np array to pyarrow array
            if isinstance(data, np.ndarray):
                out = numpy_to_pyarrow_list_array(data)
            elif isinstance(data, list) and data and isinstance(first_non_null_non_empty_value(data)[1], np.ndarray):
                out = list_of_np_array_to_pyarrow_list_array(data)
            else:
                trying_cast_to_python_objects = True
                out = pa.array(prepare_for_pa_cache(data, keep_dim=False))
            # use smaller integer precisions if possible
            if self.trying_int_optimization:
                if pa.types.is_int64(out.type):
                    out = out.cast(optimized_int_pa_type)
                elif pa.types.is_list(out.type):
                    if pa.types.is_int64(out.type.value_type):
                        out = array_cast(out, pa.list_(optimized_int_pa_type))
                    elif pa.types.is_list(out.type.value_type) and pa.types.is_int64(out.type.value_type.value_type):
                        out = array_cast(out, pa.list_(pa.list_(optimized_int_pa_type)))
            # otherwise we can finally use the user's type
            elif type is not None:
                # We use cast_array_to_feature to support casting to custom types like Audio and Image
                # Also, when trying type "string", we don't want to convert integers or floats to "string".
                # We only do it if trying_type is False - since this is what the user asks for.
                out = cast_pa_array_using_schema(
                    out, type, allow_primitive_to_str=not self.trying_type, allow_decimal_to_str=not self.trying_type
                )
            return out
        except (
                TypeError,
                pa.lib.ArrowInvalid,
                pa.lib.ArrowNotImplementedError,
        ) as e:  # handle type errors and overflows
            # Ignore ArrowNotImplementedError caused by trying type, otherwise re-raise
            if not self.trying_type and isinstance(e, pa.lib.ArrowNotImplementedError):
                raise

            if self.trying_type:
                try:  # second chance
                    if isinstance(data, np.ndarray):
                        return numpy_to_pyarrow_list_array(data)
                    elif isinstance(data, list) and data and any(isinstance(value, np.ndarray) for value in data):
                        return list_of_np_array_to_pyarrow_list_array(data)
                    else:
                        trying_cast_to_python_objects = True
                        return pa.array(prepare_for_pa_cache(data, keep_dim=False))
                except pa.lib.ArrowInvalid as e:
                    if "overflow" in str(e):
                        raise OverflowError(
                            f"There was an overflow with type {type_(data)}. Try to reduce writer_batch_size to have batches smaller than 2GB.\n({e})"
                        ) from None
                    elif self.trying_int_optimization and "not in range" in str(e):
                        optimized_int_pa_type_str = np.dtype(optimized_int_pa_type.to_pandas_dtype()).name
                        logger.info(
                            f"Failed to cast a sequence to {optimized_int_pa_type_str}. Falling back to int64."
                        )
                        return out
                    elif trying_cast_to_python_objects and "Could not convert" in str(e):
                        out = pa.array(
                            prepare_for_pa_cache(data, keep_dim=False, only_check_first_element=False)
                        )
                        if type is not None:
                            out = cast_pa_array_using_schema(
                                out, type, allow_primitive_to_str=True, allow_decimal_to_str=True
                            )
                        return out
                    else:
                        raise
            elif "overflow" in str(e):
                raise OverflowError(
                    f"There was an overflow with type {type_(data)}. Try to reduce writer_batch_size to have batches smaller than 2GB.\n({e})"
                ) from None
            elif self.trying_int_optimization and "not in range" in str(e):
                optimized_int_pa_type_str = np.dtype(optimized_int_pa_type.to_pandas_dtype()).name
                logger.info(f"Failed to cast a sequence to {optimized_int_pa_type_str}. Falling back to int64.")
                return out
            elif trying_cast_to_python_objects and "Could not convert" in str(e):
                out = pa.array(prepare_for_pa_cache(data, keep_dim=False, only_check_first_element=False))
                if type is not None:
                    out = cast_pa_array_using_schema(out, type, allow_primitive_to_str=True, allow_decimal_to_str=True)
                return out
            else:
                raise


class OptimizedTypedSequence(TypedSequence):
    def __init__(
            self,
            data,
            type: Optional[SchemaType] = None,
            try_type: Optional[SchemaType] = None,
            col: Optional[str] = None,
            optimized_int_type: Optional[SchemaType] = None,
    ):
        optimized_int_type_by_col = {
            "attention_mask": Value("int8"),  # binary tensor
            "special_tokens_mask": Value("int8"),
            "input_ids": Value("int32"),  # typical vocab size: 0-50k (max ~500k, never > 1M)
            "token_type_ids": Value(
                "int8"
            ),  # binary mask; some (XLNetModel) use an additional token represented by a 2
        }
        if type is None and try_type is None:
            optimized_int_type = optimized_int_type_by_col.get(col, None)
        super().__init__(data, type=type, try_type=try_type, optimized_int_type=optimized_int_type)


class ArrowWriter:
    """Shuffles and writes Examples to Arrow files."""

    _WRITER_CLASS = pa.RecordBatchStreamWriter

    def __init__(
            self,
            pa_schema: Optional[pa.Schema] = None,
            schema: Optional[Schema] = None,
            path: Optional[str] = None,
            stream: Optional[pa.NativeFile] = None,
            fingerprint: Optional[str] = None,
            writer_batch_size: Optional[int] = None,
            hash_salt: Optional[str] = None,
            check_duplicates: Optional[bool] = False,
            disable_nullable: bool = False,
            update_features: bool = False,
            with_metadata: bool = True,
            unit: str = "examples",
            embed_local_files: bool = False,
            storage_options: Optional[dict] = None,
    ):
        if path is None and stream is None:
            raise ValueError("At least one of path and stream must be provided.")
        if schema is not None:
            self._schema = schema
            self._pa_schema = None
        elif pa_schema is not None:
            self._pa_schema: pa.Schema = pa_schema
            self._schema = Schema.from_arrow_schema(self._pa_schema)
        else:
            self._schema = None
            self._pa_schema = None

        if hash_salt is not None:
            # Create KeyHasher instance using split name as hash salt
            self._hasher = KeyHasher(hash_salt)
        else:
            self._hasher = KeyHasher("Framework_salt")

        self._check_duplicates = check_duplicates
        self._disable_nullable = disable_nullable

        if stream is None:
            fs, path = url_to_fs(path, **(storage_options or {}))
            self._fs: fsspec.AbstractFileSystem = fs
            self._path = path if not is_remote_filesystem(self._fs) else self._fs.unstrip_protocol(path)
            self.stream = self._fs.open(path, "wb")
            self._closable_stream = True
        else:
            self._fs = None
            self._path = None
            self.stream = stream
            self._closable_stream = False

        self.fingerprint = fingerprint
        self.disable_nullable = disable_nullable
        self.writer_batch_size = (
                writer_batch_size or get_writer_batch_size(self._schema) or dataset_config.DEFAULT_MAX_BATCH_SIZE
        )
        self.update_features = update_features
        self.with_metadata = with_metadata
        self.unit = unit
        self.embed_local_files = embed_local_files

        self._num_examples = 0
        self._num_bytes = 0
        self.current_examples: list[tuple[dict[str, Any], str]] = []
        self.current_rows: list[pa.Table] = []
        self.pa_writer: Optional[pa.RecordBatchStreamWriter] = None
        self.hkey_record = []

    def __len__(self):
        """Return the number of writed and staged examples"""
        return self._num_examples + len(self.current_examples) + len(self.current_rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        # Try closing if opened; if closed: pyarrow.lib.ArrowInvalid: Invalid operation on closed file
        if self.pa_writer:  # it might be None
            try:
                self.pa_writer.close()
            except Exception:  # pyarrow.lib.ArrowInvalid, OSError
                pass
        if self._closable_stream and not self.stream.closed:
            self.stream.close()  # This also closes self.pa_writer if it is opened

    def _build_writer(self, inferred_schema: pa.Schema):
        schema = self.schema
        inferred_features = Schema.from_arrow_schema(inferred_schema)
        if self._schema is not None:
            if self.update_features:  # keep original features if they match, or update them
                fields = {field.name: field for field in self._schema.type}
                for inferred_field in inferred_features.type:
                    name = inferred_field.name
                    if name in fields:
                        if inferred_field == fields[name]:
                            inferred_features[name] = self._schema[name]
                self._schema = inferred_features
                schema: pa.Schema = inferred_schema
        else:
            self._schema = inferred_features
            schema: pa.Schema = inferred_features.arrow_schema
        if self.disable_nullable:
            schema = pa.schema(pa.field(field.name, field.type, nullable=False) for field in schema)
        if self.with_metadata:
            schema = schema.with_metadata(self._build_metadata(DatasetInfo(schema=self._schema), self.fingerprint))
        else:
            schema = schema.with_metadata({})
        self._pa_schema = schema
        self.pa_writer = self._WRITER_CLASS(self.stream, schema)

    @property
    def schema(self):
        _schema = (
            self._pa_schema
            if self._pa_schema is not None
            else (pa.schema(self._schema.type) if self._schema is not None else None)
        )
        if self._disable_nullable and _schema is not None:
            _schema = pa.schema(pa.field(field.name, field.type, nullable=False) for field in _schema)
        return _schema if _schema is not None else []

    @staticmethod
    def _build_metadata(info: DatasetInfo, fingerprint: Optional[str] = None) -> dict[str, str]:
        info_keys = ["schema"]  # we can add support for more DatasetInfo keys in the future
        info_as_dict = as_dict(info)
        metadata = {}
        metadata["info"] = {key: info_as_dict[key] for key in info_keys}
        if fingerprint is not None:
            metadata["fingerprint"] = fingerprint
        return {"CLTrainingFramework": json.dumps(metadata)}

    def write_examples_on_file(self):
        """Write stored examples from the write-pool of examples. It makes a table out of the examples and write it."""
        if not self.current_examples:
            return
        # preserve the order the columns
        if self.schema:
            schema_cols = set(self.schema.names)
            examples_cols = self.current_examples[0][0].keys()  # .keys() preserves the order (unlike set)
            common_cols = [col for col in self.schema.names if col in examples_cols]
            extra_cols = [col for col in examples_cols if col not in schema_cols]
            cols = common_cols + extra_cols
        else:
            cols = list(self.current_examples[0][0])
        batch_examples = {}
        for col in cols:
            # We use row[0][col] since current_examples contains (example, key) tuples.
            # Moreover, examples could be Arrow arrays of 1 element.
            # This can happen in `.map()` when we want to re-write the same Arrow data
            if all(isinstance(row[0][col], (pa.Array, pa.ChunkedArray)) for row in self.current_examples):
                arrays = [row[0][col] for row in self.current_examples]
                arrays = [
                    chunk
                    for array in arrays
                    for chunk in (array.chunks if isinstance(array, pa.ChunkedArray) else [array])
                ]
                batch_examples[col] = pa.concat_arrays(arrays)
            else:
                batch_examples[col] = [
                    row[0][col].to_pylist()[0] if isinstance(row[0][col], (pa.Array, pa.ChunkedArray)) else row[0][col]
                    for row in self.current_examples
                ]
        self.write_batch(batch_examples=batch_examples)
        self.current_examples = []

    def write_rows_on_file(self):
        """Write stored rows from the write-pool of rows. It concatenates the single-row tables and it writes the resulting table."""
        if not self.current_rows:
            return
        table = pa.concat_tables(self.current_rows)
        self.write_table(table)
        self.current_rows = []

    def write(
            self,
            example: dict[str, Any],
            key: Optional[Union[str, int, bytes]] = None,
            writer_batch_size: Optional[int] = None,
    ):
        """Add a given (Example,Key) pair to the write-pool of examples which is written to file.

        Args:
            example: the Example to add.
            key: Optional, a unique identifier(str, int or bytes) associated with each example
        """
        # Utilize the keys and duplicate checking when `self._check_duplicates` is passed True
        if self._check_duplicates:
            # Create unique hash from key and store as (key, example) pairs
            hash = self._hasher.hash(key)
            self.current_examples.append((example, hash))
            # Maintain record of keys and their respective hashes for checking duplicates
            self.hkey_record.append((hash, key))
        else:
            # Store example as a tuple so as to keep the structure of `self.current_examples` uniform
            self.current_examples.append((example, ""))

        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if writer_batch_size is not None and len(self.current_examples) >= writer_batch_size:
            if self._check_duplicates:
                self.check_duplicate_keys()
                # Re-initializing to empty list for next batch
                self.hkey_record = []

            self.write_examples_on_file()

    def check_duplicate_keys(self):
        """Raises error if duplicates found in a batch"""
        tmp_record = set()
        for hash, key in self.hkey_record:
            if hash in tmp_record:
                duplicate_key_indices = [
                    str(self._num_examples + index)
                    for index, (duplicate_hash, _) in enumerate(self.hkey_record)
                    if duplicate_hash == hash
                ]

                raise DuplicatedKeysError(key, duplicate_key_indices)
            else:
                tmp_record.add(hash)

    def write_row(self, row: pa.Table, writer_batch_size: Optional[int] = None):
        """Add a given single-row Table to the write-pool of rows which is written to file.

        Args:
            row: the row to add.
        """
        if len(row) != 1:
            raise ValueError(f"Only single-row pyarrow tables are allowed but got table with {len(row)} rows.")
        self.current_rows.append(row)
        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if writer_batch_size is not None and len(self.current_rows) >= writer_batch_size:
            self.write_rows_on_file()

    def write_batch(
            self,
            batch_examples: dict[str, list],
            writer_batch_size: Optional[int] = None,
            try_original_type: Optional[bool] = True,
    ):
        """Write a batch of Example to file.
        Ignores the batch if it appears to be empty,
        preventing a potential schema update of unknown types.

        Args:
            batch_examples: the batch of examples to add.
            try_original_type: use `try_type` when instantiating OptimizedTypedSequence if `True`, otherwise `try_type = None`.
        """
        if batch_examples and len(next(iter(batch_examples.values()))) == 0:
            return
        features = None if self.pa_writer is None and self.update_features else self._schema
        try_features = self._schema if self.pa_writer is None and self.update_features else None
        arrays = []
        inferred_features = Schema()
        # preserve the order the columns
        if self.schema:
            schema_cols = set(self.schema.names)
            batch_cols = batch_examples.keys()  # .keys() preserves the order (unlike set)
            common_cols = [col for col in self.schema.names if col in batch_cols]
            extra_cols = [col for col in batch_cols if col not in schema_cols]
            cols = common_cols + extra_cols
        else:
            cols = list(batch_examples)
        for col in cols:
            col_values = batch_examples[col]
            col_type = features[col] if features else None
            if isinstance(col_values, (pa.Array, pa.ChunkedArray)):
                array = cast_pa_array_using_schema(col_values, col_type) if col_type is not None else col_values
                arrays.append(array)
                inferred_features[col] = pyarrow_to_schema(col_values.type)
            else:
                col_try_type = (
                    try_features[col]
                    if try_features is not None and col in try_features and try_original_type
                    else None
                )
                typed_sequence = OptimizedTypedSequence(col_values, type=col_type, try_type=col_try_type, col=col)
                arrays.append(pa.array(typed_sequence))
                inferred_features[col] = typed_sequence.get_inferred_type()
        schema = inferred_features.arrow_schema if self.pa_writer is None else self.schema
        pa_table = pa.Table.from_arrays(arrays, schema=schema)
        self.write_table(pa_table, writer_batch_size)

    def write_table(self, pa_table: pa.Table, writer_batch_size: Optional[int] = None):
        """Write a Table to file.

        Args:
            example: the Table to add.
        """
        if writer_batch_size is None:
            writer_batch_size = self.writer_batch_size
        if self.pa_writer is None:
            self._build_writer(inferred_schema=pa_table.schema)
        pa_table = pa_table.combine_chunks()
        pa_table = pa_table_cast(pa_table, self._pa_schema)
        if self.embed_local_files:
            pa_table = for_storage(pa_table)
        self._num_bytes += pa_table.nbytes
        self._num_examples += pa_table.num_rows
        self.pa_writer.write_table(pa_table, writer_batch_size)

    def finalize(self, close_stream=True):
        self.write_rows_on_file()
        # In case current_examples < writer_batch_size, but user uses finalize()
        if self._check_duplicates:
            self.check_duplicate_keys()
            # Re-initializing to empty list for next batch
            self.hkey_record = []
        self.write_examples_on_file()
        # If schema is known, infer features even if no examples were written
        if self.pa_writer is None and self.schema:
            self._build_writer(self.schema)
        if self.pa_writer is not None:
            self.pa_writer.close()
            self.pa_writer = None
            if close_stream:
                self.stream.close()
        else:
            if close_stream:
                self.stream.close()
            raise SchemaInferenceError("Please pass `features` or at least one example when writing data")
        logger.debug(
            f"Done writing {self._num_examples} {self.unit} in {self._num_bytes} bytes {self._path if self._path else ''}."
        )
        return self._num_examples, self._num_bytes


class ParquetWriter(ArrowWriter):
    _WRITER_CLASS = pq.ParquetWriter
