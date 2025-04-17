from collections.abc import Sequence
from typing import List, Optional, Tuple, Iterable

from numpy import array, unique, ndarray


def filter_images(
    dataset: Sequence,
    labels: List,
    labels_old: List,
    background_label: List = (0, 255),
    overlap: bool = True,
) -> List:
    """
    Filter images according to the overlap.
    create a list of images and mask paths which is use for filtering.
    Also used to create the index_list.txt from tasks in config

    Parameters
    ----------
    dataset
    labels
    labels_old
    background_label
    overlap

    Returns
    -------

    """
    cls: ndarray
    index = []
    labels_cum = labels + labels_old + background_label
    if overlap:
        fil = lambda c: any(x in labels for x in cls)
    else:
        # 这里说的是：只要图片中包含有新类别就好
        fil = lambda c: any(x in labels for x in cls) and all(
            x in labels_cum for x in c
        )
    for i in range(len(dataset)):
        cls = dataset[i]["data"][1]
        cls = unique(array(cls))
        if fil(cls):
            index.append((dataset[i]["path"][0], dataset[i]["path"][1]))
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    return index


from multiprocessing import Pool
from functools import partial


def filter_images_parallel(
    dataset: Sequence,
    labels: List,
    labels_old: List,
    background_label: Iterable = (0, 255),
    overlap: bool = True,
    num_processes: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Parallel version of filter_images that processes the dataset in parallel.

    Parameters
    ----------
    dataset : Sequence
        Input dataset containing image and mask paths and data
    labels : List
        List of new labels to filter by
    labels_old : List
        List of old labels
    background_label : List, optional
        Background labels, by default (0, 255)
    overlap : bool, optional
        Whether to use overlap filtering mode, by default True
    num_processes : int, optional
        Number of processes to use, by default None (uses all available CPUs)

    Returns
    -------
    List[Tuple[str, str]]
        List of filtered image and mask path tuples
    """
    # Combine all labels for the non-overlap case
    labels_cum = labels + labels_old + list(background_label)

    # Define the filtering function based on overlap mode
    if overlap:

        def filter_func(cls):
            return any(x in labels for x in cls)

    else:

        def filter_func(cls):
            return any(x in labels for x in cls) and all(x in labels_cum for x in cls)

    # Create a partial function with the filtering criteria
    process_item = partial(_process_dataset_item, filter_func=filter_func)

    # Use multiprocessing Pool to parallelize the processing
    with Pool(processes=num_processes) as pool:
        results = pool.imap(process_item, dataset)

        # Collect results with progress reporting
        index = []
        for i, result in enumerate(results):
            if result is not None:
                index.append(result)
            if i % 1000 == 0:
                print(f"\t{i}/{len(dataset)} ...")

    return index


def _process_dataset_item(item, filter_func):
    """
    Helper function to process a single dataset item.

    Parameters
    ----------
    item : dict
        Dataset item containing 'data' and 'path' keys
    filter_func : callable
        Filtering function to apply

    Returns
    -------
    Tuple[str, str] or None
        Path tuple if item passes filter, None otherwise
    """
    cls = item["data"][1]
    cls = unique(array(cls))
    if filter_func(cls):
        return (item["path"][0], item["path"][1])
    return None


def save_list_from_filter(index, save_path):
    with open(save_path, "x") as f:
        for pair in index:
            f.write(f"{pair[0]},{pair[1]}\n")


def load_list_from_path(index, save_path):
    new_list = []
    with open(save_path, "r") as f:
        for line in f:
            x = line.split(",")
            new_list.append((x[0], x[1]))


from typing import Union, Callable, Any, List, Tuple, Generator
import numpy as np
from PIL import Image
import torch
from functools import wraps


def deep_convert_to_tensor(
    *, exclude_keys: List[str] = None, exclude_vars: List[str] = None
) -> Callable:
    """
    深度遍历转换修饰器，将嵌套结构中的所有ndarray或Image转换为torch.Tensor

    参数:
        exclude_keys: 要排除的字典键名列表
        exclude_vars: 要排除的变量名列表

    返回:
        修饰器函数
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
