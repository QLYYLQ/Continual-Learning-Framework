from numpy import array, unique
from typing import Set, List
from collections.abc import Sequence


def filter_images(dataset: Sequence, labels: List, labels_old: List, background_label: List = (0, 255),
                  overlap: bool = True) -> List:
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
    dataset.is_filter = True
    index = []
    labels_cum = labels + labels_old + background_label
    if overlap:
        fil = lambda c: any(x in labels for x in cls)
    else:
        # 这里说的是：只要图片中包含有新类别就好
        fil = lambda c: any(x in labels for x in cls) and all(x in labels_cum for x in c)
    for i in range(len(dataset)):
        cls = dataset[i]["data"][1]
        cls = unique(array(cls))
        if fil(cls):
            index.append((dataset[i]["path"][0], dataset[i]["path"][1]))
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    return index


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
