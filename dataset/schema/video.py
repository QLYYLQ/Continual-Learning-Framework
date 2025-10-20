import os
from dataclasses import dataclass, field
from typing import Optional, ClassVar, Any, Union, TYPE_CHECKING

import numpy as np
import pyarrow as pa
from torchvision.io import VideoReader

from CLTrainingFramework.dataset.arrow_utils import array_cast
if TYPE_CHECKING:
    from CLTrainingFramework.dataset.schema._type import SchemaType
from CLTrainingFramework.dataset.schema.supported_schema_type import Value
from CLTrainingFramework.dataset.schema.utils import encode_np_array, encode_torchvision_video
from CLTrainingFramework.dataset.utils.file_utils_mine import is_local_path
from CLTrainingFramework.io import io_builder, IOProtocol

_videoio: IOProtocol = io_builder(modality="Video")

@dataclass
class Video:
    load_from_storage: bool = True
    id: Optional[str] = None
    dtype: ClassVar[str] = "torchvision.io.VideoReader"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _schema: str = field(default="Video", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def sample_to_storage(self, sample):
        """
        Encode example into a format for Arrow.

        Args:
            sample (`str`, `np.ndarray`, `VideoReader` or `dict`):
                Data passed as input to Video feature.

        Returns:
            `dict` with "path" and "bytes" fields
        """

        if isinstance(sample, list):
            sample = np.array(sample)

        if isinstance(sample, str):
            return {"path": sample, "bytes": None}
        elif isinstance(sample, (bytes, bytearray)):
            return {"path": None, "bytes": sample}
        elif isinstance(sample, np.ndarray):
            # convert the video array to bytes
            return encode_np_array(sample)
        elif VideoReader is not None and isinstance(sample, VideoReader):
            # convert the torchvision video reader to bytes
            return encode_torchvision_video(sample)
        elif isinstance(sample, dict):
            path, bytes_ = sample.get("path"), sample.get("bytes")
            if path is not None and os.path.isfile(path):
                # we set "bytes": None to not duplicate the data if they're already available locally
                return {"bytes": None, "path": path}
            elif bytes_ is not None or path is not None:
                # store the video bytes, and path is used to infer the video format using the file extension
                return {"bytes": bytes_, "path": path}
            else:
                raise ValueError(
                    f"A video sample should have one of 'path' or 'bytes' but they are missing or None in {sample}."
                )
        else:
            raise TypeError(f"Unsupported encode_example type: {type(sample)}")

    def sample_from_storage(self, value: dict, token_per_repo_id=None):
        """
        bad usage:
            videos = [xxx.sample_from_storage(sample) for sample in samples]
            may cause too many open file error
        """
        if not self.load_from_storage:
            raise RuntimeError(
                f"Loading from local file is disabled for this feature. Please use Video(load_from_storage=True) instead.")
        if token_per_repo_id is None:
            token_per_repo_id = {}
        if isinstance(value, str):
            path, bytes_ = value, None
        else:
            path, bytes_ = value.get("path"), value.get("bytes")
        if bytes_ is None:
            if path is None:
                raise ValueError(
                    f"Not meaningful information in this value:{value}, should have at least path or bytes")
            elif is_local_path(path):
                video = _videoio.load(path)
            else:
                raise NotImplementedError(f"哥哥，我还没做从url下载的功能，求放过")
        else:
            video = _videoio.load(bytes_)
        video._CLTF2storage = {"path": path, "bytes": bytes_}
        return video

    def flatten(self) -> Union["SchemaType", dict[str, "SchemaType"]]:
        return (
            self
            if self.load_from_storage
            else {
                "bytes": Value("binary"),
                "path": Value("string")
            }
        )

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray, pa.ListArray]) -> pa.StructArray:

        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_list(storage.type):
            bytes_array = pa.array(
                [encode_np_array(np.array(arr))["bytes"] if arr is not None else None for arr in storage.to_pylist()],
                type=pa.binary(),
            )
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null()
            )
        return array_cast(storage, self.pa_type)
