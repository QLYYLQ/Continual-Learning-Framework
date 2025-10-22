import copy
import dataclasses
import json
import posixpath
from dataclasses import dataclass
from typing import Optional, Union, ClassVar

import fsspec
from fsspec.core import url_to_fs

from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.dataset_splits import SplitDict
from CLTrainingFramework.dataset.schema import Schema
from CLTrainingFramework.dataset.utils import path_config
from CLTrainingFramework.dataset.utils.logging import get_logger
from CLTrainingFramework.dataset.utils.py_utils_mine import as_dict, unique_values
from CLTrainingFramework.dataset.utils.version_helper import Version

logger = get_logger(__name__)


@dataclass
class SupervisedKeysData:
    input: str = ""
    output: str = ""


@dataclass
class PostProcessedInfo:
    schema: Optional[Schema] = None
    resources_checksums: Optional[dict] = None

    def __post_init__(self):
        # Convert back to the correct classes when we reload from dict
        if self.schema is not None and not isinstance(self.schema, Schema):
            self.schema = Schema.from_dict(self.schema)

    @classmethod
    def from_dict(cls, post_processed_info_dict: dict) -> "PostProcessedInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in post_processed_info_dict.items() if k in field_names})


@dataclass
class DatasetInfo:
    """Information about a dataset.

    `DatasetInfo` documents datasets, including its name, version, and features.
    See the constructor arguments and properties for a full list.

    Not all fields are known on construction and may be updated later.

    Attributes:
        description (`str`):
            A description of the dataset.
        license (`str`):
            The dataset's license. It can be the name of the license or a paragraph containing the terms of the license.
        schema ([`Schema`], *optional*):
            The schema used to specify the dataset's column types.
        post_processed (`PostProcessedInfo`, *optional*):
            Information regarding the resources of a possible post-processing of a dataset. For example, it can contain the information of an index.
        post_processing_size (`int`, *optional*):
            Size of the dataset in bytes after post-processing, if any.
        supervised_keys (`SupervisedKeysData`, *optional*):
            Specifies the input feature and the label for supervised learning if applicable for the dataset (legacy from TFDS).
        builder_name (`str`, *optional*):
            The name of the `GeneratorBasedBuilder` subclass used to create the dataset. Usually matched to the corresponding script name. It is also the snake_case version of the dataset builder class name.
        config_name (`str`, *optional*):
            The name of the configuration derived from [`BuilderConfig`].
        version (`str` or [`Version`], *optional*):
            The version of the dataset.
        splits (`dict`, *optional*):
            The mapping between split name and metadata.
        dataset_size (`int`, *optional*):
            The combined size in bytes of the Arrow tables for all splits.
        size_in_bytes (`int`, *optional*):
            The combined size in bytes of all files associated with the dataset (downloaded files + Arrow files).
        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the [`BuilderConfig`] and used in the [`DatasetBuilder`].
    """

    # Set in the dataset scripts
    description: str = dataclasses.field(default_factory=str)
    license: str = dataclasses.field(default_factory=str)
    schema: Optional[Schema] = None
    post_processed: Optional[Union[dict, PostProcessedInfo]] = None
    supervised_keys: Optional[Union[dict, SupervisedKeysData]] = None

    # Set later by the builder
    builder_name: Optional[str] = None
    dataset_name: Optional[str] = None  # for packaged builders, to be different from builder_name
    config_name: Optional[str] = None
    version: Optional[Union[str, Version]] = None
    # Set later by `download_and_prepare`
    splits: Optional[Union[SplitDict, dict]] = None
    post_processing_size: Optional[int] = None
    dataset_size: Optional[int] = None
    size_in_bytes: Optional[int] = None

    _INCLUDED_INFO_IN_YAML: ClassVar[list[str]] = [
        "config_name",
        "download_size",
        "dataset_size",
        "schema",
        "splits",
    ]

    def __post_init__(self):
        # Convert back to the correct classes when we reload from dict
        if self.schema is not None and not isinstance(self.schema, Schema):
            self.schema = Schema.from_dict(self.schema)
        if self.post_processed is not None and not isinstance(self.post_processed, PostProcessedInfo):
            self.post_processed = PostProcessedInfo.from_dict(self.post_processed)
        if self.version is not None and not isinstance(self.version, Version):
            if isinstance(self.version, str):
                self.version = Version(self.version)
            else:
                raise ValueError("好哥哥，version要str或者Version类才行")
        if self.splits is not None and not isinstance(self.splits, SplitDict):
            self.splits = SplitDict.from_split_dict(self.splits)
        if self.supervised_keys is not None and not isinstance(self.supervised_keys, SupervisedKeysData):
            if isinstance(self.supervised_keys, (tuple, list)):
                self.supervised_keys = SupervisedKeysData(*self.supervised_keys)
            else:
                self.supervised_keys = SupervisedKeysData(**self.supervised_keys)

    def write_to_directory(self, dataset_info_dir, pretty_print=True, storage_options: Optional[dict] = None):
        """
        Write `DatasetInfo` and license (if present) as JSON files to `dataset_info_dir`.

        Args:
            dataset_info_dir (`str`):
                Destination directory.
            pretty_print (`bool`, defaults to `False`):
                If `True`, the JSON will be pretty-printed with the indent level of 4.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.9.0"/>

        """
        fs: fsspec.AbstractFileSystem
        fs, *_ = url_to_fs(dataset_info_dir, **(storage_options or {}))
        with fs.open(posixpath.join(dataset_info_dir, path_config.DATASET_INFO_FILENAME), "wb") as f:
            self._dump_info(f, pretty_print=pretty_print)
        if self.license:
            with fs.open(posixpath.join(dataset_info_dir, path_config.LICENSE_FILENAME), "wb") as f:
                self._dump_license(f)

    def _dump_info(self, file, pretty_print=False):
        """Dump info in `file` file-like object open in bytes mode (to support remote files)"""
        file.write(json.dumps(as_dict(self), indent=4 if pretty_print else None).encode("utf-8"))

    def _dump_license(self, file):
        """Dump license in `file` file-like object open in bytes mode (to support remote files)"""
        file.write(self.license.encode("utf-8"))

    @classmethod
    def from_merge(cls, dataset_infos: list["DatasetInfo"]):
        dataset_infos = [dset_info.copy() for dset_info in dataset_infos if dset_info is not None]

        if len(dataset_infos) > 0 and all(dataset_infos[0] == dset_info for dset_info in dataset_infos):
            # if all dataset_infos are equal we don't need to merge. Just return the first.
            return dataset_infos[0]

        description = "\n\n".join(unique_values(info.description for info in dataset_infos)).strip()
        license = "\n\n".join(unique_values(info.license for info in dataset_infos)).strip()
        schemas = None
        supervised_keys = None

        return cls(
            description=description,
            license=license,
            schema=schemas,
            supervised_keys=supervised_keys,
        )

    @classmethod
    def from_directory(cls, dataset_info_dir: str, storage_options: Optional[dict] = None) -> "DatasetInfo":
        """
        Create [`DatasetInfo`] from the JSON file in `dataset_info_dir`.

        This function updates all the dynamically generated fields (num_examples,
        hash, time of creation,...) of the [`DatasetInfo`].

        This will overwrite all previous metadata.

        Args:
            dataset_info_dir (`str`):
                The directory containing the metadata file. This
                should be the root directory of a specific dataset version.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

        """
        fs: fsspec.AbstractFileSystem
        fs, *_ = url_to_fs(dataset_info_dir, **(storage_options or {}))
        logger.info(f"Loading Dataset info from {dataset_info_dir}")
        if not dataset_info_dir:
            raise ValueError("Calling DatasetInfo.from_directory() with undefined dataset_info_dir.")
        with fs.open(posixpath.join(dataset_info_dir, path_config.DATASET_INFO_FILENAME), "r", encoding="utf-8") as f:
            dataset_info_dict = json.load(f)
        return cls.from_dict(dataset_info_dict)

    @classmethod
    def from_dict(cls, dataset_info_dict: dict) -> "DatasetInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dataset_info_dict.items() if k in field_names})

    def update(self, other_dataset_info: "DatasetInfo", ignore_none=True):
        self_dict = self.__dict__
        self_dict.update(
            **{
                k: copy.deepcopy(v)
                for k, v in other_dataset_info.__dict__.items()
                if (v is not None or not ignore_none)
            }
        )

    def copy(self) -> "DatasetInfo":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    def to_yaml_dict(self) -> dict:
        yaml_dict = {}
        dataset_info_dict = as_dict(self)
        for key in dataset_info_dict:
            if key in self._INCLUDED_INFO_IN_YAML:
                value = getattr(self, key)
                if hasattr(value, "to_yaml_list"):  # Features, SplitDict
                    yaml_dict[key] = value.to_yaml_list()
                elif hasattr(value, "to_yaml_string"):  # Version
                    yaml_dict[key] = value.to_yaml_string()
                else:
                    yaml_dict[key] = value
        return yaml_dict

    @classmethod
    def from_yaml_dict(cls, yaml_data: dict) -> "DatasetInfo":
        yaml_data = copy.deepcopy(yaml_data)
        if yaml_data.get("features") is not None:
            yaml_data["features"] = Schema.from_yaml_list(yaml_data["features"])
        if yaml_data.get("splits") is not None:
            yaml_data["splits"] = SplitDict.from_yaml_list(yaml_data["splits"])
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in yaml_data.items() if k in field_names})
