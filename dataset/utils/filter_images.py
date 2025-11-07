from collections.abc import Sequence
from typing import List, Optional, Tuple, Iterable

from numpy import array, unique, ndarray


def filter_images(
    dataset: Sequence,
    labels: List,
    labels_old: List,
    background_label: set = (0),
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
