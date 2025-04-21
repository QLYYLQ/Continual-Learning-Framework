from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
from PIL import Image

_change_type = [Image.Image, np.ndarray]
_target_type = [torch.Tensor]


def change_input_to_tensor(
    func: Callable[[Any], Any],
    changed_type: Optional[Iterable[type[Any]]] = None,
    target_type: Optional[Iterable[type[Any]]] = None,
):
    if changed_type is None:
        changed_type = _change_type
    if target_type is None:
        target_type = _target_type

    @wraps(func)
    def wrapper(*args, **kwargs):
        input_data = args[0]
        if isinstance(input_data, dict):
            for key in input_data.keys():
                input_data[key] = torch.from_numpy(input_data[key]).float()
        else:
            input_data = torch.from_numpy(input_data).float()
        return wrapper(input_data, *args[1:], **kwargs)


from typing import Union, Callable, Any, List, Generator
import numpy as np
from PIL import Image
import torch
from functools import wraps


def deep_convert_to_tensor(
    *, exclude_keys: Optional[List[str]], exclude_vars: Optional[List[str]]
) -> Callable:
    """
    using exclude_keys and exclude_vars to exclude some keys or variables from conversion
    automatic convert ndarray and Image(in args, kwargs, dict, Iterable object) to torch.Tensor
    """
    if exclude_keys is None:
        exclude_keys = []
    if exclude_vars is None:
        exclude_vars = []

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 处理位置参数
            new_args = tuple(
                _deep_convert(arg, exclude_keys, exclude_vars, "arg" + str(i))
                for i, arg in enumerate(args)
            )

            # 处理关键字参数
            new_kwargs = {
                k: _deep_convert(v, exclude_keys, exclude_vars, k)
                for k, v in kwargs.items()
            }

            return func(*new_args, **new_kwargs)

        return wrapper

    return decorator


def _deep_convert(
    obj: Any, exclude_keys: List[str], exclude_vars: List[str], current_name: str
) -> Any:
    """
    深度遍历转换函数

    参数:
        obj: 要转换的对象
        exclude_keys: 排除的键名列表
        exclude_vars: 排除的变量名列表
        current_name: 当前对象在结构中的名称/路径

    返回:
        转换后的对象
    """
    if current_name in exclude_vars:
        return obj

    # 使用生成器遍历所有可能的转换项
    for converted in _deep_convert_generator(obj, exclude_keys, current_name):
        return converted

    return obj


def _deep_convert_generator(
    obj: Any, exclude_keys: List[str], current_path: str
) -> Generator[Any, None, None]:
    """
    生成器函数，深度遍历并转换对象

    参数:
        obj: 要遍历的对象
        exclude_keys: 排除的键名列表
        current_path: 当前路径（用于排除检查）

    生成:
        转换后的对象
    """
    # 基础类型直接返回
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        yield obj
        return

    # 直接转换ndarray和Image
    if isinstance(obj, (np.ndarray, Image.Image)):
        yield _convert_single_input(obj)
        return

    # 处理字典（包括嵌套）
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            full_path = f"{current_path}.{k}" if current_path else k
            if k in exclude_keys:
                new_dict[k] = v
            else:
                for converted in _deep_convert_generator(v, exclude_keys, full_path):
                    new_dict[k] = converted
        yield new_dict
        return

    # 处理列表、元组、集合等可迭代对象
    if isinstance(obj, (list, tuple, set)):
        converted_items = []
        for i, item in enumerate(obj):
            full_path = f"{current_path}[{i}]"
            for converted in _deep_convert_generator(item, exclude_keys, full_path):
                converted_items.append(converted)

        # 保持原始容器类型
        if isinstance(obj, tuple):
            yield tuple(converted_items)
        elif isinstance(obj, set):
            yield set(converted_items)
        else:
            yield converted_items
        return

    # 其他类型直接返回
    yield obj


def _convert_single_input(input_data: Union[np.ndarray, Image.Image]) -> torch.Tensor:
    """转换单个输入为torch.Tensor (与之前相同)"""
    if isinstance(input_data, Image.Image):
        input_data = np.array(input_data)

    if not isinstance(input_data, np.ndarray):
        return input_data

    if input_data.dtype == np.uint8:
        input_data = input_data.astype(np.float32) / 255.0

    tensor = torch.from_numpy(input_data)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[2] in [1, 3, 4]:
            tensor = tensor.permute(2, 0, 1)

    return tensor


_converter_dict: dict[object, Callable[[Any], Any]]
_target_type: list[Any] = [torch.Tensor]


# 使用示例
@deep_convert_to_tensor(exclude_keys=["metadata"], exclude_vars=["debug_img"])
def process_data(data, debug_img=None):
    """
    这个函数现在可以接收包含嵌套结构的输入，
    其中的ndarray和Image会被自动转换为torch.Tensor

    排除:
    - 字典中键为"metadata"的项
    - 变量名为"debug_img"的参数
    """
    print("处理数据:")
    print(f"输入类型: {type(data)}")

    if isinstance(data, dict):
        print("字典内容:")
        for k, v in data.items():
            print(f"{k}: {type(v)}")

    return data


# 测试
if __name__ == "__main__":
    # 测试嵌套结构
    nested_data = {
        "image": np.random.rand(224, 224, 3),
        "mask": Image.new("L", (224, 224), 128),
        "metadata": {"info": "不应该被转换", "array": np.array([1, 2, 3])},
        "samples": [np.random.rand(64, 64), {"thumbnail": Image.new("RGB", (32, 32))}],
    }

    result = process_data(data=nested_data, debug_img=np.array([1, 2, 3]))  # 这个应该被排除
