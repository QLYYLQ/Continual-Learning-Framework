from torch.utils.data import Dataset

_dataset_name = {}


def register_training_dataset(fn: Dataset):
    """
    注册数据集，被这个函数修饰过的函数都会把所在的文件名当作数据集名存在_dataset_name中
    不用__name__的原因是dataset的类名多半不直观，不如用文件名来得简单
    """
    dataset_name_split = fn.__module__.split('.')
    dataset_name_split.append(fn.__name__)
    dataset_name = ".".join(dataset_name_split[-2:])
    _dataset_name[dataset_name] = fn
    return fn


def register_validation_dataset(fn: Dataset):
    """
    注册数据集，被这个函数修饰过的函数都会把所在的文件名当作数据集名存在_dataset_name中
    不用__name__的原因是dataset的类名多半不直观，不如用文件名来得简单
    """
    dataset_name_split = fn.__module__.split('.')
    dataset_name_split.append(fn.__name__)
    dataset_name = ".".join(dataset_name_split[-2:])
    _dataset_name[dataset_name] = fn
    return fn

def dataset_entrypoints(dataset_name: str):
    """根据名字返回对应数据集的类"""
    return _dataset_name[dataset_name]


def is_dataset_registered(dataset_name: str) -> bool:
    """根据名字确认是否注册了对应的数据集"""
    return dataset_name in _dataset_name
