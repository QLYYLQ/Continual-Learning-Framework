from typing import Union

from CLTrainingFramework.dataset.arrow_utils import Array2D, Array3D, Array4D, Array5D
from CLTrainingFramework.dataset.schema.image import Image
from CLTrainingFramework.dataset.schema.supported_schema_type import Value, ClassLabel, Sequence, LargeSequence
from CLTrainingFramework.dataset.schema.video import Video

SchemaType = Union[
    dict,
    list,
    tuple,
    Value,
    ClassLabel,
    Sequence,
    LargeSequence,
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Image,
    Video,
]
_SCHEMA_TYPES: dict[str, SchemaType] = {
    Value.__name__: Value,
    ClassLabel.__name__: ClassLabel,
    LargeSequence.__name__: LargeSequence,
    Sequence.__name__: Sequence,
    Array2D.__name__: Array2D,
    Array3D.__name__: Array3D,
    Array4D.__name__: Array4D,
    Array5D.__name__: Array5D,
    Image.__name__: Image,
    Video.__name__: Video,
}
