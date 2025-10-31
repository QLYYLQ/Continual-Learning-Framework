from dataclasses import dataclass, field, InitVar
from typing import Optional, Union, Iterable, Literal, ClassVar, Any
from typing import Sequence as TypingSequence

import pyarrow as pa
import pyarrow.compute as pc

from CLTrainingFramework.dataset.arrow_utils import array_cast
from CLTrainingFramework.dataset.arrow_utils import str_to_arrow_type


@dataclass
class Value:
    """
    Args:
        dtype(str):
            Name of the data type
    """
    dtype: str
    id: Optional[str] = None
    pa_type:ClassVar[Any] = None
    _schema: str = field(default="Value", init=False, repr=False)

    def __post_init__(self):
        if self.dtype == "double":
            self.dtype = "float64"
        if self.dtype == "float":
            self.dtype = "float32"
        if self.dtype == "int":
            self.dtype = "int32"
        self.pa_type = str_to_arrow_type(self.dtype)

    def __call__(self):
        return self.pa_type

    def sample_to_pa_cache(self, value):
        if pa.types.is_boolean(self.pa_type):
            return bool(value)
        elif pa.types.is_integer(self.pa_type):
            return int(value)
        elif pa.types.is_floating(self.pa_type):
            return float(value)
        elif pa.types.is_string(self.pa_type):
            return str(value)
        else:
            return value


@dataclass
class ClassLabel:
    names: list[str] = None
    id: Optional[str] = None
    dtype: ClassVar[str] = "int64"
    pa_type: pa.DataType = pa.int64()
    _schema: str = field(default="ClassLabel", init=False, repr=False)
    # 避免在__eq__等方法中比较这两个属性
    names_file: InitVar[Optional[str]] = None
    names_num: InitVar[Optional[int]] = None
    mode: InitVar[Literal["strict", "capital_insensitive"]] = field(default="strict", repr=False)

    def __post_init__(self, names_file: Optional[str], names_num: Optional[int],
                      mode: Literal["strict", "capital_insensitive"]):
        self.names_file = names_file
        self.names_num = names_num
        self.mode = mode
        if self.names_num is not None and self.names_file is not None:
            raise ValueError("names_file and names_num are mutually exclusive")
        if self.names is None:
            if self.names_file is not None:
                self.names = self._load_from_file(self.names_file)
            elif self.names_num is not None:
                self.names = [str(i) for i in range(self.names_num)]
            else:
                raise ValueError("Please provide names or names_num or names_file")
        elif not isinstance(self.names, TypingSequence):
            raise TypeError(f"names must be a sequence, but get {type(self.names)}")
        if self.names_num is None:
            self.names_num = len(self.names)
        elif self.names_num != len(self.names):
            raise ValueError("names_num and names must have same length")
        self._int2str = [str(name) for name in self.names]
        self._str2int = {name: i for i, name in enumerate(self.names)}
        self._lower_str2int = {name.lower(): i for i, name in enumerate(self.names)}
        if len(self._int2str) != len(self._str2int):
            raise ValueError("some label names are duplicated")
    def __call__(self):
        return self.pa_type
    def int2str(self, values: Union[int, Iterable]) -> Union[Iterable[str], str]:
        if not isinstance(values, int) and not isinstance(values, Iterable):
            raise TypeError(f"values must be int or Iterable, but get {type(values)}")
        _list = True
        if isinstance(values, int):
            values = [values]
            _list = False
        output = []
        for v in values:
            if not 0 <= v < self.names_num:
                raise ValueError(f"out of range, get index {v}, but only have {self.names_num} labels")
            output.append(self._int2str[int(v)])
        return output if _list else output[0]

    def str2int(self, values: Union[str, Iterable[str]]) -> int:
        if not isinstance(values, str) and not isinstance(values, Iterable):
            raise TypeError(f"values must be str or Iterable, but get {type(values)}")
        _list = True
        if isinstance(values, str):
            values = [values]
            _list = False
        output = [self._str_val2int(value) for value in values]
        return output if _list else output[0]

    @staticmethod
    def _load_from_file(names_file):
        with open(names_file, encoding="utf-8") as f:
            # 避免空换行
            return [name.strip() for name in f.read().split("\n") if name.strip()]

    def _str_val2int(self, value: str) -> int:
        _fail = False
        value = str(value)
        int_return = self._str2int.get(value)
        if int_return is None:
            if self.mode == "strict":
                int_return = self._str2int.get(value.strip())
            elif self.mode == "capital_insensitive":
                int_return = self._lower_str2int.get(value.lower())
            else:
                raise ValueError(f"Unknown mode {self.mode}")
            if int_return is None:
                try:
                    int_value = int(value)
                except ValueError:
                    _fail = True
                else:
                    if int_value < -1 or int_value >= self.names_num:
                        _fail = True
        if _fail:
            raise ValueError(f"invalid string class lable {value}")
        return int_return

    def sample_to_pa_cache(self, sample):
        if self.names_num is None:
            raise ValueError("Please provide names or names_num")
        if isinstance(sample, str):
            sample = self.str2int(sample)
        if not -1 <= sample < self.names_num:
            raise ValueError(f"invalid string class lable {sample}, total length is {self.names_num}")
        return sample

    def prepare_for_pa_cache(self, storage: Union[pa.StringArray, pa.IntegerArray]) -> pa.Int64Array:
        if isinstance(storage, pa.IntegerArray) and len(storage) > 0:
            min_max = pc.min_max(storage).as_py()
            if min_max["max"] is not None and min_max["max"] >= self.names_num:
                raise ValueError(
                    f"Class label {min_max['max']} greater than configured names number {self.names_num}"
                )
        elif isinstance(storage, pa.StringArray):
            storage = pa.array(
                [
                    self._str_val2int(label) if label is not None else None
                    for label in storage.to_pylist()
                ]
            )
        return array_cast(storage, self.pa_type)


@dataclass
class Sequence:
    """
    Construct a list of feature from a single type or a dict of types.
    Mostly here for compatiblity with tfds.

    Args:
        schema ([`SchemaType`]):
            A list of features of a single type or a dictionary of types.
        length (`int`):
            Length of the sequence.
    """

    schema: Any
    length: int = -1
    id: Optional[str] = None
    # Automatically constructed
    dtype: ClassVar[str] = "list"
    pa_type: ClassVar[Any] = None
    _schema: str = field(default="Sequence", init=False, repr=False)


@dataclass
class LargeSequence:
    """Feature type for large list data composed of child feature data type.

    It is backed by `pyarrow.LargeListType`, which is like `pyarrow.ListType` but with 64-bit rather than 32-bit offsets.

    Args:
        schema ([`FeatureType`]):
            Child feature data type of each item within the large list.
    """

    schema: Any
    id: Optional[str] = None
    # Automatically constructed
    pa_type: ClassVar[Any] = None
    _schema: str = field(default="LargeSequence", init=False, repr=False)


if __name__ == '__main__':
    a = Value(dtype="int")
    print(a)
    print(a._schema)
    b = ClassLabel(names_file="test")
    c = ClassLabel(names_num=2)
    print(b == c)
