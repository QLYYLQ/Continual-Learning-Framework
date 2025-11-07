# 文本比图片复杂一些，因为图片对编码的要求全被包在PIL里了
import json
from typing import Any

import yaml  # type: ignore

from CLTrainingFramework.io.Protocol import _StrOrBytesPath
from CLTrainingFramework.io.Register import create_io_registry

TextIOMeta = create_io_registry("Text")


class BaseText(metaclass=TextIOMeta):  # type: ignore
    suffixes = ["txt"]

    def load(
            self, path: _StrOrBytesPath, mode: str = "r", encoding: str = "utf-8"
    ) -> Any:
        with open(path, mode=mode, encoding=encoding) as f:  # type: ignore
            text = f.read()
        return text

    def write(
            self, path: _StrOrBytesPath, text: str, mode: str = "w", encoding: str = "utf-8"
    ) -> None:
        with open(path, mode=mode, encoding=encoding) as f:  # type: ignore
            f.write(text)


class JsonText(BaseText):
    suffixes = ["json"]

    def load(
            self, path: _StrOrBytesPath, mode: str = "r", encoding: str = "utf-8"
    ) -> Any:
        with open(path, mode, encoding=encoding) as f:  # type: ignore
            json_data = json.load(f)
        return json_data

    def write(
            self, path: _StrOrBytesPath, text: str, mode: str = "w", encoding: str = "utf-8"
    ) -> None:
        try:
            with open(path, mode=mode, encoding=encoding) as f:  # type: ignore
                # support chinese
                json.dump(text, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Can't write json file: {e}")
        except TypeError as e:
            print(f"JSON serialization error: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")


class YamlText(BaseText):
    suffixes = ["yaml"]

    def load(
            self, path: _StrOrBytesPath, mode: str = "r", encoding: str = "utf-8"
    ) -> Any:
        with open(path, mode=mode, encoding=encoding) as f:  # type: ignore
            data = yaml.safe_load(f)
        return data

    def write(
            self, path: _StrOrBytesPath, text: str, mode: str = "w", encoding: str = "utf-8"
    ) -> None:
        try:
            with open(path, mode=mode, encoding=encoding) as f:  # type: ignore
                yaml.dump(text, f, allow_unicode=True, indent=4)
        except IOError as e:
            print(f"Can't write yaml file: {e}")
        except TypeError as e:
            print(f"YAML serialization error: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")
