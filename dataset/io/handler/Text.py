# 文本比图片复杂一些，因为图片对编码的要求全被包在PIL里了

from dataset.io.Protocol import _StrOrBytesPath
from dataset.io.Register import TextIOMeta


class BaseText(metaclass=TextIOMeta):  # type: ignore
    suffixes = ["txt"]

    def load(self, path: _StrOrBytesPath, mode: str = "r", encoding: str = "utf-8"):
        with open(path, mode, encoding) as f:  # type: ignore
            text = f.read()
        return text

    def write(
        self, path: _StrOrBytesPath, text: str, mode: str = "w", encoding: str = "utf-8"
    ):
        with open(path, mode, encoding) as f:  # type: ignore
            f.write(text)


class JsonText(BaseText):
    suffixes = ["json"]
