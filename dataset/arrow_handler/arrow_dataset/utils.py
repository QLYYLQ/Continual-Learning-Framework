import json
import os
import shutil
import tempfile
import weakref
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

import pyarrow as pa

from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.dataset_info import DatasetInfo
from CLTrainingFramework.dataset.arrow_handler.arrow_table import Table, MemoryTable
from CLTrainingFramework.dataset.arrow_handler.arrow_writer import ArrowWriter
from CLTrainingFramework.dataset.schema import Schema
from CLTrainingFramework.dataset.utils import path_config
from CLTrainingFramework.dataset.utils.py_utils_mine import as_dict

if TYPE_CHECKING:
    from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.arrow_dataset import Dataset


def _check_table(table) -> Table:
    """We check the table type to make sure it's an instance of :class:`datasets.table.Table`"""
    if isinstance(table, pa.Table):
        # for a pyarrow table, we can just consider it as a in-memory table
        # this is here for backward compatibility
        return MemoryTable(table)
    elif isinstance(table, Table):
        return table
    else:
        raise TypeError(f"Expected a pyarrow.Table or a datasets.table.Table object, but got {table}.")


# ------------------------ cache utils
_USING_CACHE: bool = True
_TEMP_DIR_FOR_CACHE: Optional["_TempCacheDir"] = None
_DATASETS_WITH_TABLE_IN_TEMP_DIR: Optional[weakref.WeakSet] = None


def get_datasets_with_cache_file_in_temp_dir():
    return list(_DATASETS_WITH_TABLE_IN_TEMP_DIR) if _DATASETS_WITH_TABLE_IN_TEMP_DIR is not None else []


class _TempCacheDir:
    """
    A temporary directory for storing cached Arrow files with a cleanup that frees references to the Arrow files
    before deleting the directory itself to avoid permission errors on Windows.
    """

    def __init__(self):
        self.name = tempfile.mkdtemp(prefix=path_config.TEMP_CACHE_DIR_PREFIX)
        self._finalizer = weakref.finalize(self, self._cleanup)

    def _cleanup(self):
        for dset in get_datasets_with_cache_file_in_temp_dir():
            dset.__del__()
        if os.path.exists(self.name):
            try:
                shutil.rmtree(self.name)
            except Exception as e:
                raise OSError(
                    f"An error occurred while trying to delete temporary cache directory {self.name}. Please delete it manually."
                ) from e

    def cleanup(self):
        if self._finalizer.detach():
            self._cleanup()


def register_dataset_for_no_cache(dataset):
    if _TEMP_DIR_FOR_CACHE is None:
        return
    global _DATASETS_WITH_TABLE_IN_TEMP_DIR
    if _DATASETS_WITH_TABLE_IN_TEMP_DIR is None:
        _DATASETS_WITH_TABLE_IN_TEMP_DIR = weakref.WeakSet()
    if any(
            Path(_TEMP_DIR_FOR_CACHE.name) in Path(i["filename"]).parents
            for i in dataset.cache_files
    ):
        _DATASETS_WITH_TABLE_IN_TEMP_DIR.add(dataset)


def _using_cache():
    global _USING_CACHE
    return bool(_USING_CACHE)


def get_temp_cache_dir():
    global _TEMP_DIR_FOR_CACHE
    if _TEMP_DIR_FOR_CACHE is None:
        _TEMP_DIR_FOR_CACHE = _TempCacheDir()
    return _TEMP_DIR_FOR_CACHE.name

# -------------------------

def _check_column_names(column_names: list[str]):
    """Check the column names to make sure they don't contain duplicates."""
    counter = Counter(column_names)
    if not all(count == 1 for count in counter.values()):
        duplicated_columns = [col for col in counter if counter[col] > 1]
        raise ValueError(f"The table can't have duplicated columns but columns {duplicated_columns} are duplicated.")


def update_metadata_with_schema(table: Table, features: Schema):
    """
    To be used in dataset transforms that modify the features of the dataset, in order to update the features stored in the metadata of its schema.
    """
    features = Schema({col_name: features[col_name] for col_name in table.column_names})
    if table.schema.metadata is None or b"CLTrainingFramework" not in table.schema.metadata:
        pa_metadata = ArrowWriter._build_metadata(DatasetInfo(schema=features))
    else:
        metadata = json.loads(table.schema.metadata[b"CLTrainingFramework"].decode())
        if "info" not in metadata:
            metadata["info"] = as_dict(DatasetInfo(schema=features))
        else:
            metadata["info"]["schema"] = as_dict(DatasetInfo(schema=features))["schema"]
        pa_metadata = {"CLTrainingFramework": json.dumps(metadata)}
    table = table.replace_schema_metadata(pa_metadata)
    return table


# ----fingerprint helper------

def transmit_format(func):
    """Wrapper for dataset transforms that recreate a new Dataset to transmit the format of the original dataset to the new dataset"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: "Dataset" = args[0]
            args = args[1:]
        else:
            self: "Dataset" = kwargs.pop("self")
        # don't use self.format since it returns a list of columns for 'columns' even if self_format_columns is None
        unformatted_columns = set(self.column_names) - set(self._format_columns or [])
        self_format = {
            "type": self._format_type,
            "format_kwargs": self._format_kwargs,
            "columns": self._format_columns,
            "output_all_columns": self._output_all_columns,
        }
        # apply actual function
        out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
        datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
        # re-apply format to the output
        for dataset in datasets:
            new_format = self_format.copy()
            if new_format["columns"] is not None:  # new formatted columns = (columns - previously unformatted columns)
                # sort the columns to have a deterministic list of columns that we can compare with `out_format`
                new_format["columns"] = sorted(set(dataset.column_names) - unformatted_columns)
            out_format = {
                "type": dataset._format_type,
                "format_kwargs": dataset._format_kwargs,
                "columns": sorted(dataset._format_columns) if dataset._format_columns is not None else None,
                "output_all_columns": dataset._output_all_columns,
            }
            if out_format != new_format:
                fingerprint = dataset._fingerprint
                dataset.set_format(**new_format)
                dataset._fingerprint = fingerprint
        return out

    wrapper._decorator_name_ = "transmit_format"
    return wrapper
