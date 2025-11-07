from typing import Optional, TYPE_CHECKING

from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.dataset_info import DatasetInfo
from CLTrainingFramework.dataset.arrow_handler.arrow_dataset.dataset_splits import NamedSplit
if TYPE_CHECKING:
    from CLTrainingFramework.dataset.schema import Schema


class DatasetInfoPlugin:
    """This base class exposes some attributes of DatasetInfo
    at the base level of the Dataset for easy access.
    """

    def __init__(self, info: DatasetInfo, split: Optional[NamedSplit]):
        self._info = info
        self._split = split

    @property
    def info(self):
        """[`~datasets.DatasetInfo`] object containing all the metadata in the dataset."""
        return self._info

    @property
    def split(self):
        """[`~datasets.NamedSplit`] object corresponding to a named dataset split."""
        return self._split

    @property
    def builder_name(self) -> str:
        return self._info.builder_name

    @property
    def config_name(self) -> str:
        return self._info.config_name

    @property
    def dataset_size(self) -> Optional[int]:
        return self._info.dataset_size

    @property
    def description(self) -> str:
        return self._info.description

    @property
    def schema(self) -> Optional[Schema]:
        return self._info.schema.copy() if self._info.schema is not None else None

    @property
    def license(self) -> Optional[str]:
        return self._info.license

    @property
    def size_in_bytes(self) -> Optional[int]:
        return self._info.size_in_bytes

    @property
    def supervised_keys(self):
        return self._info.supervised_keys

    @property
    def version(self):
        return self._info.version


