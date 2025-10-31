import os.path
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional, ClassVar, Any, Union, TYPE_CHECKING

import PIL
import numpy as np
import pyarrow as pa
from PIL import ImageOps
from PIL.Image import Image as img

from CLTrainingFramework.dataset.arrow_utils import array_cast
if TYPE_CHECKING:
    from CLTrainingFramework.dataset.schema._type import SchemaType
from CLTrainingFramework.dataset.schema.utils import encode_pil_image, encode_np_array
from CLTrainingFramework.dataset.utils.file_utils_mine import is_local_path
from CLTrainingFramework.dataset.utils.py_utils_mine import no_op_if_value_is_null
from CLTrainingFramework.io import io_builder, IOProtocol



_imageio: IOProtocol = io_builder(modality="Image")
@dataclass
class Image:
    mode: Optional[str] = None
    load_from_storage: bool = True
    id: Optional[str] = None
    dtype: ClassVar[str] = "Image"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _schema: str = field(default="Image", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def sample_to_storage(self, value: Union[str, bytes, bytearray, dict, np.array, img]) -> dict:
        if isinstance(value, list):
            value = np.array(value)
        if isinstance(value, str):
            return {"path": value, "bytes": None}
        elif isinstance(value, (bytes, bytearray)):
            return {"path": None, "bytes": value}
        elif isinstance(value, np.ndarray):
            return encode_np_array(value)
        elif isinstance(value, img):
            return encode_pil_image(value)
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            return {"path": value["path"], "bytes": None}
        elif value.get("bytes") is not None or value.get("path") is not None:
            return {"path": value.get("path"), "bytes": value.get("bytes")}
        else:
            raise ValueError(f"An image sample should have bytes or bytearray, but got {value}")

    def sample_from_storage(self, value: dict, token_per_repo_id=None) -> dict:
        if not self.load_from_storage:
            raise RuntimeError(
                "Loading from local file is disabled for this feature. Please use Image(load_from_storage=True) instead."
            )
        if token_per_repo_id is None:
            token_per_repo_id = {}
        path, bytes_ = value["path"], value["bytes"]
        if bytes_ is None:
            if path is None:
                raise ValueError(
                    f"An image should have one of path or bytes but both are None in {value}"
                )
            else:
                if is_local_path(path):
                    kwargs = value.get("kwargs", {})
                    image = _imageio.load(path, **kwargs)
                else:
                    raise NotImplementedError("哥，还没做从url开始下载image的功能，求放过")
        else:
            image = PIL.Image.open(BytesIO(bytes_))
        if image.getexif().get(PIL.Image.ExifTags.Base.Orientation) is not None:
            image = ImageOps.exif_transpose(image)
        if self.mode and self.mode != image.mode:
            image = image.convert(self.mode)
        return image

    def flatten(self) -> Union["SchemaType", dict[str, "SchemaType"]]:
        from CLTrainingFramework.dataset.schema.supported_schema_type import Value
        return (
            self
            if self.load_from_storage
            else
            {
                "bytes": Value("binary"),
                "path": Value("path")
            }
        )

    def prepare_for_arrow_storage(self, storage: Union[pa.StringArray, pa.StructArray, pa.ListArray]) -> pa.StructArray:
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays(
                [bytes_array, storage], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [storage, path_array], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_list(storage.type):
            bytes_array = pa.array(
                [
                    encode_np_array(np.array(arr))["bytes"] if arr is not None else None
                    for arr in storage.to_pylist()
                ],
                type=pa.binary(),
            )
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null()
            )
        return array_cast(storage, self.pa_type)

    def embed_storage(self, storage: pa.StructArray) -> pa.StructArray:
        """Embed image files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.

        Returns:
            `pa.StructArray`: Array in the Image arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """

        @no_op_if_value_is_null
        def path_to_bytes(path):
            with open(path, "rb") as f:
                bytes_ = f.read()
            return bytes_

        bytes_array = pa.array(
            [
                (path_to_bytes(x["path"]) if x["bytes"] is None else x["bytes"])
                if x is not None
                else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [
                os.path.basename(path) if path is not None else None
                for path in storage.field("path").to_pylist()
            ],
            type=pa.string(),
        )
        storage = pa.StructArray.from_arrays(
            [bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null()
        )
        return array_cast(storage, self.pa_type)
