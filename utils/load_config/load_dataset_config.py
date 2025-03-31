import os.path as osp
import random
from pathlib import Path
from utils.load_config.base_load import get_config
from munch import Munch
import pickle
from copy import deepcopy

dataset_config_root_path = Path(__file__).resolve().parent.parent.parent.joinpath('config', 'dataset')
dataset_root_path = Path(__file__).resolve().parent.parent.parent.joinpath("data")


def get_dataset_config(dataset_name):
    config_path = dataset_config_root_path.joinpath(dataset_name + '.yaml')
    config = get_config(config_path)
    template_config = _load_template_config()
    _check_config(config, template_config)
    config = _modify_dataset_config(config)
    return config


def _load_template_config():
    config_path = dataset_config_root_path.joinpath('template.yaml')
    if not osp.exists(config_path):
        raise FileNotFoundError("please check the file, template is missing")
    config = get_config(config_path)
    return config


def _modify_dataset_config(config_transient):
    """change default setting in config"""
    config_transient.training = _modify_training(config_transient.training, config_transient.class_number)
    config_transient.dataset_setting = _modify_dataset_setting(config_transient.dataset_setting, config_transient.name)
    config_transient.increment_setting = _modify_increment_setting(config_transient.increment_setting, config_transient.name)
    return config_transient


def _modify_training(config_transient, class_number):
    copied_config = deepcopy(config_transient)
    class_index = [x for x in range(class_number)]
    for task, task_setting in config_transient.items():
        ignore_index = config_transient[task].ignore_index
        class_number1 = class_number-len(ignore_index)
        design = [int(x) for x in task_setting.design.split('-')]
        if sum(design)+sum(ignore_index) != class_number1:
            raise ValueError("please check the design. total number is not equal to class number")
        # 排除ignore_index以后的类别
        label_list = [x for x in class_index if x not in ignore_index]
        stage_number = {x: y for x, y in enumerate(design)}
        index_order = {}
        if task_setting.index_order == "random":
            for key, number in stage_number.items():
                index_order[key], label_list = _random_select_and_remove(label_list, number)
        elif task_setting.index_order == "default":
            last_number = 0
            for key, number in stage_number.items():
                number += last_number
                index_order[key] = label_list[last_number:number]
                last_number = number
        else:
            task_setting.index_order = dict(task_setting.index_order)
        copied_config[task].index_order = Munch.fromDict(index_order)
    return copied_config


def _modify_dataset_setting(config_transient, dataset_name):
    classes_path = dataset_config_root_path.joinpath(config_transient.classes)
    # 修改default的classes name路径
    if config_transient.classes == 'default':
        classes_path = dataset_config_root_path.joinpath('classes', f"{dataset_name}_classes.pkl")
    if not osp.exists(classes_path):
        raise FileNotFoundError("please check the file, classes_name is missing")
    with open(classes_path, 'rb') as f:
        classes = pickle.load(f)
    config_transient.classes = classes
    dataset_root = dataset_root_path.joinpath(config_transient.root)
    if not osp.exists(dataset_root):
        raise FileNotFoundError(f"please check the file, dataset root is missing. the path is {dataset_root}")
    config_transient.root = str(dataset_root)
    if getattr(config_transient,"image_size") is not None:
        image_size = config_transient.image_size
        numbers = image_size.strip("()").split(",")
        image_size = tuple(int(x) for x in numbers)
        config_transient.image_size = image_size
    return config_transient


def _modify_increment_setting(config_transient, dataset_name):
    if config_transient.increment_dataset_name == "default":
        config_transient.increment_dataset_name = dataset_name+".Increment"
    if config_transient.split_dataset_name == "default":
        config_transient.split_dataset_name = dataset_name + ".Split"
    if config_transient.save_stage_image_path == "default":
        #这里在后续加载任务中实现，不是在这里实现
        pass
    return config_transient


# def _check_config(config_transient, template_config):
#     for setting, _ in template_config.items():
#         if not hasattr(config_transient, setting):
#             raise AttributeError(f"please check the config, {setting} is missing")
#     if not hasattr(config_transient.training, "task_1"):
#         raise AttributeError("must design task_1 at first")
#
#     for task, task_setting in config_transient.training.items():
#         for setting, _ in template_config.training.task_1.items():
#             if not hasattr(task_setting, setting):
#                 raise AttributeError(f"please check the config, in {task}, the setting: {setting} is missing")
#     for task, task_setting in config_transient.evaluate.items():
#         for setting, _ in template_config.evaluate.task_1.items():
#             if not hasattr(task_setting, setting):
#                 raise AttributeError(f"please check the config, in {task}, the setting: {setting} is missing")
#     for task, _ in config_transient.evaluate.items():
#         if not hasattr(config_transient.training, task):
#             raise AttributeError(f"please check the config, evaluate and training should have same task {task}")
#     for setting, _ in template_config.dataset_setting.items():
#         if not hasattr(config_transient.dataset_setting, setting):
#             raise AttributeError(f"please check the config, you should have dataset setting: {setting}")
#     for setting, _ in template_config.increment_setting.items():
#         if not hasattr(config_transient.increment_setting, setting):
#             raise AttributeError(f"please check the config, you should have increment dataset setting: {setting}")

def _check_config(config_transient, template_config):
    base_setting = set(config_transient.keys())
    base_template_setting = set(template_config.keys())
    unsettings = base_template_setting - base_setting
    if unsettings:
        raise ValueError(f"please check the template, you have something unsetting in your config which is: {unsettings}")
    for template_setting in base_template_setting:
        if isinstance(template_config[template_setting],Munch):
            _check_config(config_transient[template_setting], template_config[template_setting])






def _random_select_and_remove(number_list, n):
    selected = random.sample(number_list, n)
    for num in selected:
        number_list.remove(num)
    return selected, number_list


if __name__ == '__main__':
    config = get_dataset_config('VOC')
    print(config)
    # config = get_dataset_config("ade")
