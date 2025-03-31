import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Union, Dict, List
from functools import wraps
from torch import Tensor
def hex_to_rgba(hex_color, alpha=1.0):
    """将十六进制颜色转换为RGBA格式（值范围在0-1之间）"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
    return np.array([*rgb, alpha])

# 定义颜色列表





def check_input(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(image, masks, *args, **kwargs):
        if isinstance(image, Tensor):
            image = image.cpu().detach().numpy()
            image = np.transpose(image, axes=[1, 2, 0])
        if isinstance(masks, Tensor):
            masks = masks.cpu().detach().numpy()
        if kwargs.get("type",None) is None:
            raise ValueError("must specify type")
        else:
            if kwargs["type"] not in ["origin", "separated","binary"]:
                raise ValueError("type must be either 'origin' or 'separated'")
        if (kwargs.get("points", None) is not None) and (kwargs.get("bboxs", None) is not None):
            # if is origin label's bbox
            if isinstance(kwargs["bboxs"], np.ndarray):
                if kwargs["bboxs"].shape[0] != kwargs["points"].shape[0]:
                    raise ValueError("Both points and bboxs must have the same shape")
            if isinstance(kwargs["bboxs"], list):
                if len(kwargs["bboxs"]) != kwargs["points"].shape[0]:
                    raise ValueError("Both points and bboxs must have the same shape")
        return fn(image, masks, *args, **kwargs)

    return wrapper


def check_color(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(**kwargs):
        if kwargs.get("color_info") is not None:
            if len(kwargs["color_info"]) != 4:
                raise ValueError("color_info must have 4 elements")
        return fn(**kwargs)

    return wrapper


def create_color_list(color_number) -> np.ndarray:
    color = []
    for i in range(color_number):
        color.append(np.concatenate([np.random.random(3), np.array([0.7])]))
    return np.stack(color)


def show_mask(mask, ax, random_color=False,color_info=None):
    """mask is a true false mask with shape:[H,W]"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    if color_info is not None:
        if len(color_info) != 4:
            raise ValueError("color_info must have 4 elements, with rgb and transparency in [0,1]")
    h, w = mask.shape[-2:]
    if color_info is None:
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    else:
        mask_image = mask.reshape(h, w, 1) * color_info.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, label, ax,color_info, marker_size=100):
    edge_color = ["red","green"]
    color = color_info[:-1]
    ax.scatter([coord[0] for coord in coords], [coord[1] for coord in coords], color=color, marker='*', s=marker_size, edgecolor=edge_color[label],
               linewidth=1.25)



def show_box(box, ax, color_info, number=None):
    if color_info is None:
        color_info = [0,1.0,0,1]
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    edge_color = color_info[:3]
    alpha = color_info[3]
    ax.add_patch(plt.Rectangle((x0+4, y0+4), w-8, h-8, edgecolor=edge_color, facecolor=(0,0,0,0), lw=2))
    if number is not None:
        ax.text(x0+8, y0+20, str(number), color=edge_color, fontweight='bold', fontsize='small')


@check_input
def show_image_with_mask_and_prompt(image,masks, file_name, foreground_points=None,background_points=None, bboxs=None, mask_value = [0],type=None):
    """
    masks has the shape with the shape: [1,h,w] or [labels_number,1,h,w]
    points with the shape: [batch_size,points_number,2]
    bbox with the shape:[batch_size,bbox_number,1,4]
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if type == "origin":
        for mask in masks:
            mask = mask.squeeze()
            value_list = np.unique(mask)
            value_list = [x for x in value_list if x not in mask_value]
            color_list = create_color_list(len(value_list))
            for index,value in enumerate(value_list):
                mask1 = mask.copy()
                mask1[mask == value] = True
                mask1[mask != value] = False
                show_mask(mask1, plt.gca(), False,color_list[index])
                if bboxs is not None:
                    bbox = bboxs[index]
                    show_box(bbox, plt.gca(), color_list[index],index)
                if foreground_points is not None:
                    foreground_point = foreground_points[index]
                    background_point = background_points[index]
                    show_points(foreground_point,1,plt.gca(),color_list[index])
                    show_points(background_point,0,plt.gca(),color_list[index])
    elif type == "separated":
        color_list = create_color_list(masks.shape[0])
        for index,mask in enumerate(masks):
            mask = mask.squeeze()
            value_list = np.unique(mask)
            value_list = [x for x in value_list if x not in mask_value]
            for value in value_list:
                mask1 = mask.copy()
                mask1[mask == value] = True
                mask1[mask != value] = False
                show_mask(mask1, plt.gca(), False, color_list[index])
                bbox = bboxs[index]
                if bbox is not None:
                    for bbox1 in bbox:
                        show_box(bbox1, plt.gca(), color_list[index],index)
                foreground_point = foreground_points[index]
                background_point = background_points[index]
                if foreground_point is not None:
                    for point in foreground_point:
                        show_points(point,1,plt.gca(),color_list[index])
                if background_point is not None:
                    for point in background_point:
                        show_points(point,0,plt.gca(),color_list[index])

    plt.axis('off')
    plt.savefig(file_name + '_' + "origin" + '.png', bbox_inches='tight', pad_inches=-0.1)
    plt.close()

def change_for_cuda_tensor(fn:Callable)->Callable:
    def change(input):
        if isinstance(input,np.ndarray):
            return input
        elif isinstance(input,torch.Tensor):
            input = input.cpu().detach().numpy()
            return input
        else:
            return input
        
    @wraps(fn)
    def wrapps(*args,**kwargs):
        for arg in args:
            if isinstance(arg,list):
                for dict1 in arg:
                    for k,v in dict1.items():
                        dict1[k] = change(dict1[k])
        return fn(*args,**kwargs)


def separate_points_from_points_list(points_list:Union[torch.Tensor,np.ndarray],points_label:Union[torch.Tensor,np.ndarray]):
    if points_list.shape[0:2] != points_label.shape[0:2]:
        raise ValueError("points_list must have the same shape with points label: [1,num]")
    foreground_points_list = []
    background_points_list = []
    for i in range(points_label.shape[1]):
        if points_label[0,i] == 1:
            a = points_list[0,i,:].cpu().detach().numpy()
            foreground_points_list.append(a)
        else:
            a = points_list[0,i,:].cpu().detach().numpy()
            background_points_list.append(a)
    return foreground_points_list,background_points_list





def show_picture_with_return_dict(return_dict_list:List[Dict[str,torch.Tensor]],file_name,color_input=None,label_list=None,type=None):
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    if not isinstance(file_name,str):
        file_name = str(file_name)
    image = return_dict_list[0]["image"]
    for index,return_dict in enumerate(return_dict_list):
        if not torch.equal(image,return_dict["image"]):
            raise ValueError(f"in this list, has some mask for other image, the image in {index} is different from the first image")
    image = image.detach().cpu().numpy()
    image = np.transpose(image, axes=[1, 2, 0])
    if type is None:
        image = (image*std)+mean
    image = np.clip(image,0,255).astype(np.uint8)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    if color_input is None:
        color_list = create_color_list(len(return_dict_list))
    else:
        color_list = color_input
    for index,return_dict in enumerate(return_dict_list):
        
        if label_list is not None:
            index = label_list[index]
        if return_dict["label"] is not None:
            show_mask(return_dict["label"],plt.gca(),False,color_list[index])
        if return_dict["point_coords"] is not None:
            points1,points2 = separate_points_from_points_list(return_dict["point_coords"],return_dict["point_labels"])
            show_points(points1,1,plt.gca(),color_list[index])
            show_points(points2,0,plt.gca(),color_list[index])
        box = return_dict["boxes"]
        if box is not None:
            box = box[0]
            box = box.cpu().detach().numpy()
            show_box(box,plt.gca(),color_list[index],index)
    plt.axis('off')
    plt.savefig(file_name+ '.png', bbox_inches='tight', pad_inches=-0.1)
    plt.close()








# def NewShowImage(image,masks, file_name, points = None,points_label=None, bboxs=None, mask_value = [0],type=None):
#     """satisfy for sam input(binary mask in list and points with points label)"""
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     if type == "origin":
#         for mask in masks:
#             mask = mask.squeeze()
#             value_list = np.unique(mask)
#             value_list = [x for x in value_list if x not in mask_value]
#             color_list = create_color_list(len(value_list))
#             for index,value in enumerate(value_list):
#                 mask1 = mask.copy()
#                 mask1[mask == value] = True
#                 mask1[mask != value] = False
#                 show_mask(mask1, plt.gca(), False,color_list[index])
#                 if bboxs is not None:
#                     bbox = bboxs[index]
#                     show_box(bbox, plt.gca(), color_list[index],index)
#                 if foreground_points is not None:
#                     foreground_point = foreground_points[index]
#                     background_point = background_points[index]
#                     show_points(foreground_point,1,plt.gca(),color_list[index])
#                     show_points(background_point,0,plt.gca(),color_list[index])
#     elif type == "separated":
#         color_list = create_color_list(masks.shape[0])
#         for index,mask in enumerate(masks):
#             mask = mask.squeeze()
#             value_list = np.unique(mask)
#             value_list = [x for x in value_list if x not in mask_value]
#             for value in value_list:
#                 mask1 = mask.copy()
#                 mask1[mask == value] = True
#                 mask1[mask != value] = False
#                 show_mask(mask1, plt.gca(), False, color_list[index])
#                 bbox = bboxs[index]
#                 if bbox is not None:
#                     for bbox1 in bbox:
#                         show_box(bbox1, plt.gca(), color_list[index],index)
#                 foreground_point = foreground_points[index]
#                 background_point = background_points[index]
#                 if foreground_point is not None:
#                     for point in foreground_point:
#                         show_points(point,1,plt.gca(),color_list[index])
#                 if background_point is not None:
#                     for point in background_point:
#                         show_points(point,0,plt.gca(),color_list[index])
#     plt.axis('off')
#     plt.savefig(file_name + '_' + "origin" + '.png', bbox_inches='tight', pad_inches=-0.1)
#     plt.close()

def voc_cmap(N=21, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 4), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b,255])

    cmap = cmap/255 if normalized else cmap
    return cmap


import colorsys
import random


def generate_color_palette(base_colors=None, n: int = 15):
    # 将十六进制转换为RGB格式（0-1范围）
    if base_colors is None:
        base_colors = ['#F2E4D8', '#F2AFA0', '#F2220F', '#F2594B', '#F28379']

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

    # 将RGB转换为十六进制
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

    # 转换基础颜色为RGB
    base_rgb = [hex_to_rgb(color) for color in base_colors]
    new_colors = []

    for i in range(n):
        if i < len(base_rgb):
            # 直接使用基础颜色
            new_colors.append(base_rgb[i])
        else:
            # 随机选择两个基础颜色进行插值
            color1 = random.choice(base_rgb)
            color2 = random.choice(base_rgb)
            ratio = random.random()

            # 生成新颜色
            new_color = tuple(
                color1[j] * ratio + color2[j] * (1 - ratio)
                for j in range(3)
            )

            # 添加一些随机变化
            new_color = tuple(
                min(1.0, max(0.0, c + (random.random() - 0.5) * 0.1))
                for c in new_color
            )
            new_color = list(new_color).append(1)
            new_colors.append(new_color)

    # 转换回十六进制格式
    hex_colors = [rgb_to_hex(color) for color in new_colors]
    # 转换为RGB格式（0-255范围）
    rgb_colors = [(int(r * 255), int(g * 255), int(b * 255))
                  for r, g, b in new_colors]

    return rgb_colors

if __name__ == '__main__':
    # color_palette = generate_color_palette(n=15)
    # print(color_palette)
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    colors = voc_cmap(N=21, normalized=True)

    # 创建画布
    plt.figure(figsize=(12, 2))

    # 绘制色块
    for i in range(21):
        plt.bar(i, 1, color=colors[i, :3], width=1)

    # 设置图表属性
    plt.xlim(-0.5, 20.5)
    plt.ylim(0, 1)
    plt.title('VOC Colormap 颜色展示')
    plt.xticks(range(21))
    plt.grid(False)

    plt.show()

