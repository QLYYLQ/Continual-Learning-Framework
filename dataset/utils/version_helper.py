import dataclasses
import re
from functools import total_ordering
from dataclasses import dataclass
from typing import Optional, Union

from typing_extensions import Tuple

_VERSION_REG = re.compile(r"^(?P<major>\d+)" r"\.(?P<minor>\d+)" r"\.(?P<patch>\d+)$")


@total_ordering
@dataclass
class Version:
    """
    Version class
    Args:
        version_str
        description
        major
        minor
        patch
    """

    version_str: str
    description: Optional[str] = None
    major: Optional[Union[int, str]] = None
    minor: Optional[Union[int, str]] = None
    patch: Optional[Union[int, str]] = None

    def __post_init__(self):
        self.major, self.minor, self.patch = Version._str_to_version(self.version_str)

    def _validate_operand(self, other):
        if isinstance(other, str):
            return Version(other)
        elif isinstance(other, Version):
            return other
        raise TypeError(
            f"{other} (type: {type(other)}) can't be converted to Version.(str only)"
        )

    @staticmethod
    def _str_to_version(version_str: str):
        res = _VERSION_REG.match(version_str)
        if not res:
            raise ValueError(
                f"{version_str} is not a valid version string, format should be 'major.minor.patch'"
            )
        return tuple(
            int(v) for v in [res.group("major"), res.group("minor"), res.group("patch")]
        )

    @staticmethod
    def _tuple_to_str(version_tuple):
        return ".".join(str(v) for v in version_tuple)

    @property
    def tuple(self):
        return self.major, self.minor, self.patch

    def __repr__(self):
        return f"{self.tuple[0]}.{self.tuple[1]}.{self.tuple[2]}"

    def __eq__(self, other):
        try:
            other = self._validate_operand(other)
        except (TypeError, ValueError):
            return False
        else:
            return self.tuple == other.tuple

    def __lt__(self, other):
        other = self._validate_operand(other)
        return self.tuple < other.tuple

    def __hash__(self):
        return hash(Version._tuple_to_str(self.tuple))
    @classmethod
    def from_dict(cls,dic:dict):
        field_name = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k:v for k,v in dic.items() if k in field_name})
    def to_yaml_string(self):
        return self.version_str


if __name__ == "__main__":
    version1 = Version("1.0.0")
    version2 = Version("1.0.1")
    print(version2 > version1)
    print(version2 < version1)
    print(version1.tuple)
