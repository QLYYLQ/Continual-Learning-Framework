import numpy as np
from scipy.ndimage import label, measurements
from typing import Optional, Dict, List, Union, Tuple, Any, Callable
from torch import Tensor
from functools import wraps


def preprocess_input_labels(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(input_labels: Union[np.ndarray, Tensor], mask_value: Optional[List[int]] = None, *args, **kwargs):
        if isinstance(input_labels, Tensor):
            input_labels = input_labels.cpu().numpy()
        else:
            input_labels = input_labels.copy()
        if mask_value is None:
            mask_value = [0]
        height, width = input_labels.shape[-2], input_labels.shape[-1]
        return func(input_labels, mask_value=mask_value, height=height, width=width, *args, **kwargs)

    return wrapper


def find_separated_boundaries(input_labels: Union[Tensor, np.ndarray], label_list: Optional[List[List[int]]] = None,
                              min_size_percentage: int = 0.05, max_size_percentage: int = 100) -> Tuple[
    List[Dict[int, Union[int, Any]]], List[Any]]:
    """
    找出每一块闭合的轮廓。

    参数:
    mask : Optional[Tensor,np.ndarray]
        shape:[B,1,H,W]
    label_list : Optional[List[List[int]]]
        list of labels in each mask, can be none
    min_size_percentage : int
        when mask size smaller than min_size_percentage% times total size, ignore it

    返回:
    Union[List[Dict[int,np.array]],List[np.array]]
        Dict
            batch中每个mask对应的label_mask和它们的细分遮罩
        List
            这些遮罩堆叠的结果
    """

    if type(input_labels) == Tensor:
        input_labels = input_labels[:,0,:,:].cpu().numpy()

    if label_list is None:
        label_list = np.unique(input_labels)
    label_mask = []
    stacked_label_mask = []
    total_pixel = input_labels.shape[-2] * input_labels.shape[-1]
    min_size = total_pixel * min_size_percentage / 100.0
    max_size = total_pixel * max_size_percentage / 100.0
    for index in range(input_labels.shape[0]):
        transient_label = input_labels[index].copy()
        labels = label_list[index]
        transient_mask = {}
        stacked_mask = []
        for label_value in labels:
            mask1 = transient_label.copy()
            mask1[transient_label != label_value] = False
            mask1[transient_label == label_value] = True
            labeled_mask, number = label(mask1.squeeze(), structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            labeled_mask = labeled_mask[None]
            area_size = measurements.sum(mask1, labeled_mask, index=range(1, number + 1))
            for transient_index, size in enumerate(area_size):
                if size < min_size or size > max_size:
                    # 因为number是从1开始的，enumerate的迭代器是从0开始的
                    # 小遮罩不应该包括那些面积过大的部分
                    labeled_mask[labeled_mask == (transient_index + 1)] = False
            transient_mask[label_value] = labeled_mask
            stacked_mask.append(labeled_mask)
        stacked_mask = np.stack(stacked_mask)
        label_mask.append(transient_mask)
        stacked_label_mask.append(stacked_mask)

    return label_mask, stacked_label_mask


def create_bbox_from_labels(labels: np.ndarray, mask_value: Optional[List[int]] = None):
    """[b,1,H,W]的label或者是[num_labels,1,H,W]的label拆分"""
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    if mask_value is None:
        mask_value = [0]
    origin_label_bbox = []
    for origin_label in range(labels.shape[0]):
        label1 = labels[origin_label, 0, :, :].copy()
        label_value = [x for x in np.unique(label1) if x not in mask_value]
        bbox_for_single_label = []
        for label_index in label_value:
            bbox = create_bbox(label1, label_index)
            bbox_for_single_label.append(bbox)
        bbox_for_single_label = np.stack(bbox_for_single_label) if bbox_for_single_label else None
        origin_label_bbox.append(bbox_for_single_label)

    return origin_label_bbox


def create_bbox(labels: np.ndarray, value: Union[int, bool]):
    """传入的mask要是二维的"""
    labels = labels.squeeze()
    true_y, true_x = np.where(labels == value)
    return np.array([np.min(true_x), np.min(true_y), np.max(true_x), np.max(true_y)])


def create_bbox_from_origin_label(origin_label: np.ndarray):
    """create bbox for mask with shape:[batch_size,1,H,W]"""
    bbox = create_bbox_from_labels(origin_label)
    return bbox


def create_bbox_from_separated_label(separated_label: List[np.ndarray]):
    """create bbox for mask with shape:[[labels_number,1,H,W],...,] with the len equal to batch size"""
    bbox = []
    for single_label in separated_label:
        single_separated_bbox = create_bbox_from_labels(single_label)
        bbox.append(single_separated_bbox)
    return bbox


def create_bbox_from_origin_and_separated_label(origin_label: Union[Tensor, np.ndarray],
                                                separated_label: List[np.ndarray]) -> Tuple[List, List]:
    bbox = create_bbox_from_origin_label(origin_label)
    bbox1 = create_bbox_from_separated_label(separated_label)
    return bbox, bbox1


@preprocess_input_labels
def create_foreground_points_from_label_with_step(input_labels: Union[np.ndarray, Tensor], height: int, width: int,
                                                  pixel_points_ratio: int = 200, mask_value=None):
    """
    pixel_points_ratio: every 40000 pixels have a point
    mask has the shape of [batch,1,height,width] or [labels_number,1,height,width]
    """
    point_list = []
    for input_label in input_labels:
        point_dict = {}
        label_value_list = np.unique(input_label)
        label_value_list = [x for x in label_value_list if x not in mask_value]
        for label_value in label_value_list:
            transient_point = []
            mask1 = input_label.copy()
            mask1[input_label != label_value] = False
            mask1[input_label == label_value] = True
            for y in range(0, height, pixel_points_ratio):
                for x in range(0, width, pixel_points_ratio):
                    if mask1[0][y][x]:
                        transient_point.append([x, y])
            transient_point = np.stack(transient_point) if transient_point else None
            point_dict[label_value] = transient_point
        point_list.append(point_dict)
    return point_list


@preprocess_input_labels
def create_background_points_from_label_with_step(input_labels: Union[np.ndarray, Tensor], height: int, width: int,
                                                  pixel_points_ratio: int = 200,
                                                  mask_value: Optional[List[int]] = None):
    """masks can be [batch,1,h,w] or [labels_number,1,h,w]"""
    point_list = []
    for input_label in input_labels:
        point_dict = {}
        label_value_list = np.unique(input_label)
        label_value_list = [x for x in label_value_list if x not in mask_value]
        for label_value in label_value_list:
            transient_point = []
            mask1 = input_label.copy()
            mask1[input_label != label_value] = False
            mask1[input_label == label_value] = True
            for y in range(0, height, pixel_points_ratio):
                for x in range(0, width, pixel_points_ratio):
                    if not mask1[0][y][x]:
                        transient_point.append([x, y])
            transient_point = np.stack(transient_point) if transient_point else []
            point_dict[label_value] = transient_point
        point_list.append(point_dict)
    return point_list


@preprocess_input_labels
def create_points_from_label_with_number(input_labels: Union[np.ndarray, Tensor], height: int, width: int,
                                         foreground_points_number: int = 10, background_points_number: int = 10,
                                         mask_value=None):
    """
    input_labels with shape likes [batch,1,H,W] or [labels_number,1,H,W]
    """
    foreground_point_list = []
    background_point_list = []
    for input_label in input_labels:
        label_value_list = np.unique(input_label)
        label_value_list = [x for x in label_value_list if x not in mask_value]
        transient_foreground_point = []
        transient_background_point = []
        for label_value in label_value_list:
            [x_min, y_min, x_max, y_max] = create_bbox(input_label, label_value)
            cropped_labels = input_label[0, y_min:y_max + 1, x_min:x_max + 1]
            a,b = np.where(cropped_labels == label_value)
            selected_points = np.column_stack((b,a))
            a, b = np.where(cropped_labels != label_value)
            other_points = np.column_stack((b,a))
            transient_foreground_point1 = (random_select_points_from_points_list(foreground_points_number, selected_points))
            transient_background_point1 = (random_select_points_from_points_list(background_points_number, other_points))
            if transient_foreground_point1 is not None:
                transient_foreground_point1 += np.array([x_min, y_min])
            else:
                transient_foreground_point1  = np.tile(np.array([x_min,y_min]),(foreground_points_number,1))
            if transient_background_point1 is not None:
                transient_background_point1 += np.array([x_min, y_min])
            else:
                transient_background_point1 = np.tile(np.array([x_min,y_min]),(background_points_number,1))
            transient_background_point.append(transient_background_point1)
            transient_foreground_point.append(transient_foreground_point1)
        transient_foreground_point = np.array(transient_foreground_point)
        transient_background_point = np.array(transient_background_point)
        foreground_point_list.append(
            transient_foreground_point) if transient_foreground_point.size > 0 else foreground_point_list.append(None)
        background_point_list.append(
            transient_background_point) if transient_background_point.size > 0 else background_point_list.append(None)
    return foreground_point_list, background_point_list


def random_select_points_from_points_list(points_number, points_ndarray):
    if points_ndarray.size == 0:
        return None
    if points_ndarray.shape[0] < points_number:
        indices = np.random.choice(points_ndarray.shape[0], points_number, replace=True)
    else:
        indices = np.random.choice(points_ndarray.shape[0], points_number, replace=False)
    return points_ndarray[indices]


def check_point_creation(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(origin_label, separated_label, point_type, **kwargs):
        if point_type not in ["number", "step"]:
            raise ValueError("point_type must be either 'number' or 'step'")
        if point_type == "number":
            if (kwargs.get("foreground_number", None) is None) or (kwargs.get("background_number", None) is None):
                raise ValueError("foreground_number and background_number must be specified")
            else:
                return fn(origin_label, separated_label, point_type, kwargs["foreground_number"],
                          kwargs["background_number"])
        elif point_type == "step":
            if (kwargs.get("foreground_step", None) is None) or (kwargs.get("background_step", None) is None):
                raise ValueError("foreground_step and background_step must be specified")
            else:
                return fn(origin_label, separated_label, point_type, kwargs["foreground_step"],
                          kwargs["background_step"])

    return wrapper


def create_point_from_origin_label(origin_label, point_type, foreground_ratio, background_ratio):
    if point_type == "step":
        raise NotImplementedError
    elif point_type == "number":
        foreground_points1, background_point1 = create_points_from_label_with_number(origin_label,
                                                                                     foreground_points_number=foreground_ratio,
                                                                                     background_points_number=background_ratio)
        return foreground_points1, background_point1


def create_point_from_separated_label(separated_label, point_type, foreground_ratio, background_ratio):
    if point_type == "step":
        raise NotImplementedError
    elif point_type == "number":
        foreground_points, background_points = [], []
        for single_label in separated_label:
            foreground_points1, background_points1 = create_points_from_label_with_number(single_label,
                                                                                          foreground_points_number=foreground_ratio,
                                                                                          background_points_number=background_ratio)
            foreground_points.append(foreground_points1)
            background_points.append(background_points1)
        return foreground_points, background_points


@check_point_creation
def create_point_from_origin_and_separated_label(origin_label, separated_label, point_type: str, foreground_ratio,
                                                 background_ratio, ):
    """
    You can choose a type to generate point prompts. With type: number, you can generate prompt with a selected number,
    with the type: step, you can generate prompt with a selected step
    """
    origin_points1, origin_points2 = create_point_from_origin_label(origin_label, point_type, foreground_ratio,
                                                                    background_ratio)
    separated_points1, separated_points2 = create_point_from_separated_label(separated_label, point_type,
                                                                             foreground_ratio, background_ratio)
    return origin_points1, origin_points2, separated_points1, separated_points2


def create_point_label(points: np.ndarray, point_type=None):
    if point_type is None:
        point_type = [1, 0]
    foreground_shape = points.shape[:-1] + (1,)
    background_shape = points.shape[:-1] + (1,)
    return np.ones(foreground_shape), np.zeros(background_shape)


def create_point_label_from_origin_points(origin_points: List[np.ndarray]):
    batch_size = len(origin_points)
    foreground_points_labels = []
    background_points_labels = []
    for points in origin_points:
        foreground_points_label, background_points_label = create_point_label(points=points)
        foreground_points_labels.append(foreground_points_label)
        background_points_labels.append(background_points_label)
    return foreground_points_labels, background_points_labels


def create_point_label_from_separated_points(separated_points: List[List[np.ndarray]]):
    batch_size = len(separated_points)
    foreground_points_labels = []
    background_points_labels = []
    for points in separated_points:
        transient_foreground = []
        transient_background = []
        for separated_point in points:
            if separated_point is not None:
                foreground_points_label, background_points_label = create_point_label(points=separated_point)
                transient_foreground.append(foreground_points_label)
                transient_background.append(background_points_label)
            else:
                transient_foreground.append(None)
                transient_background.append(None)
        foreground_points_labels.append(transient_foreground)
        background_points_labels.append(transient_background)
    return foreground_points_labels, background_points_labels


def create_point_label_from_origin_and_separated_points(origin_points: List[np.ndarray],
                                                        separated_points: List[List[np.ndarray]]):
    points_label11, points_label12 = create_point_label_from_origin_points(origin_points)
    points_label21, points_label22 = create_point_label_from_separated_points(separated_points)
    return points_label11, points_label12, points_label21, points_label22


if __name__ == '__main__':
    masks = np.empty([3, 1, 1024, 1024])
    masks[:] = np.random.choice([1, 2, 3], size=(3, 1, 1024, 1024))
    points = create_foreground_points_from_label_with_step(masks)
