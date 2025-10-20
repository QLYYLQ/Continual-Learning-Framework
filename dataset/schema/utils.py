import sys
import warnings
from io import BytesIO
from typing import Optional, TypedDict

import PIL
import PIL.Image
import numpy as np

from CLTrainingFramework.dataset.schema.supported_schema_type import LargeSequence, Sequence


def _check_non_null_non_empty_recursive(
        obj, schema=None
) -> bool:
    """
    Check if the object is not None.
    If the object is a list or a tuple, recursively check the first element of the sequence and stop if at any point the first element is not a sequence or is an empty sequence.
    """
    if obj is None:
        return False
    elif isinstance(obj, (list, tuple)) and (
            schema is None or isinstance(schema, (list, tuple, LargeSequence, Sequence))
    ):
        if len(obj) > 0:
            if schema is None:
                pass
            elif isinstance(schema, (list, tuple)):
                schema = schema[0]
            else:
                schema = schema.schema
            return _check_non_null_non_empty_recursive(obj[0], schema)
        else:
            return False
    else:
        return True


_IMAGE_COMPRESSION_FORMATS: Optional[list[str]] = None


def _PIL_image_compression_formats() -> list[str]:
    global _IMAGE_COMPRESSION_FORMATS
    if _IMAGE_COMPRESSION_FORMATS is None:
        PIL.Image.init()
        _IMAGE_COMPRESSION_FORMATS = list(
            set(PIL.Image.OPEN.keys()) & set(PIL.Image.SAVE.keys())
        )
    return _IMAGE_COMPRESSION_FORMATS


def write_image(obj):
    # PIL.Image.Image has attr: filename which include the absolute path
    if hasattr(obj, "filename") and obj.filename != "":
        # for local image, we only remember its path
        return {"path": obj.filename, "bytes": None}
    else:
        # for online image, we show remember its bytes
        return {"path": None, "bytes": image_to_bytes(obj)}


def image_to_bytes(image: PIL.Image.Image) -> bytes:
    buffer = BytesIO()
    if image.format in _PIL_image_compression_formats():
        format = image.format
    else:
        format = "PNG" if image.mode in ["1", "L", "LA", "RGB", "RGBA"] else "TIFF"
    image.save(buffer, format=format)
    return buffer.getvalue()


def encode_pil_image(image: "PIL.Image.Image") -> dict:
    if hasattr(image, "filename") and image.filename != "":
        return {"path": image.filename, "bytes": None}
    else:
        return {"path": None, "bytes": image_to_bytes(image)}


def encode_np_array(array: np.ndarray) -> dict:
    dtype = array.dtype
    dtype_byteorder = dtype.byteorder if dtype.byteorder != "=" else _NATIVE_BYTEORDER
    dtype_kind = dtype.kind
    dtype_itemsize = dtype.itemsize

    dest_dtype = None

    if array.shape[2:]:
        if dtype_kind not in ["u", "i"]:
            raise TypeError(
                f"Unsupported array dtype {dtype} for image encoding. Only {dest_dtype} is supported for multi-channel arrays."
            )
        dest_dtype = np.dtype("|u1")
        if dtype != dest_dtype:
            warnings.warn(
                f"Downcasting array dtype {dtype} to {dest_dtype} to be compatible with 'Pillow'"
            )
    elif dtype in _VALID_IMAGE_ARRAY_DTYPES:
        dest_dtype = dtype
    else:
        while dtype_itemsize >= 1:
            dtype_str = dtype_byteorder + dtype_kind + str(dtype_itemsize)
            if np.dtype(dtype_str) in _VALID_IMAGE_ARRAY_DTYPES:
                dest_dtype = np.dtype(dtype_str)
                warnings.warn(
                    f"Downcasting array dtype {dtype} to {dest_dtype} to be compatible with 'Pillow'"
                )
                break
            else:
                dtype_itemsize //= 2
        if dest_dtype is None:
            raise TypeError(
                f"Cannot downcast dtype {dtype} to a valid image dtype. Valid image dtypes: {_VALID_IMAGE_ARRAY_DTYPES}"
            )

    image = PIL.Image.fromarray(array.astype(dest_dtype))
    return {"path": None, "bytes": image_to_bytes(image)}


class _Example(TypedDict):
    path: Optional[str]
    bytes: Optional[bytes]


def encode_torchvision_video(video: "VideoReader") -> _Example:
    if hasattr(video, "_CLTF2storage"):
        return video._CLTF2storage
    else:
        raise NotImplementedError(
            "Encoding a VideoReader that doesn't come from Video.sample_from_storage() is not implemented"
        )


_NATIVE_BYTEORDER = "<" if sys.byteorder == "little" else ">"
_VALID_IMAGE_ARRAY_DTYPES = [
    np.dtype("|b1"),
    np.dtype("|u1"),
    np.dtype("<u2"),
    np.dtype(">u2"),
    np.dtype("<i2"),
    np.dtype(">i2"),
    np.dtype("<u4"),
    np.dtype(">u4"),
    np.dtype("<i4"),
    np.dtype(">i4"),
    np.dtype("<f4"),
    np.dtype(">f4"),
    np.dtype("<f8"),
    np.dtype(">f8"),
]
