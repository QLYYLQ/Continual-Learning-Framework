import torch
import numpy as np
from functools import wraps
from typing import Callable,Union,Optional,List
def change_label_into_binary_label(input_labels: np.ndarray, mask_value=[0]):
    """input_label with the shape of [B,1,H,W]"""
    return_labels = []
    for input_label in input_labels:
        label_value_list = np.unique(input_label)
        label_value_list = [x for x in label_value_list if x not in mask_value]
        return_label = []
        for value in label_value_list:
            label1 = input_label.copy()
            label1[input_label != value] = False
            label1[input_label == value] = True
            if label1.any():
                return_label.append(label1[None])
        if return_label:
            return_label = np.concatenate(return_label, axis=0)
            return_labels.append(return_label)
    if return_labels:
        return_labels = np.concatenate(return_labels, axis=0)
    return return_labels


def modify_label(input_labels):
    reutrn_labels = []
    for input_label in input_labels:
        reutrn_labels.append(change_label_into_binary_label(input_label))
    return reutrn_labels



def modify_separated_prompts(bboxs_list, foreground_points_list, background_list):
    def remove_none(input_prompt):
        return_list = []
        for prompt in input_prompt:
            if prompt is not None:
                return_list.append(prompt)
        return return_list

    return_bbox, return_foreground_point, return_background_point = [], [], []
    for bboxs, foreground_points, background_points in zip(bboxs_list, foreground_points_list, background_list):
        bboxs = remove_none(bboxs)
        bbox_list = []
        for i in bboxs:
            bbox_list+=[i[index:index+1] for index in range(i.shape[0])]
        return_bbox.append(bbox_list)
        foreground_points =  remove_none(foreground_points)
        background_points = remove_none(background_points)
        point_list1,point_list2 = [],[]
        for points1,points2 in zip(foreground_points,background_points):
            point_list1 += [points1[i:i+1] for i in range(points1.shape[0])]
            point_list2 += [points2[i:i+1] for i in range(points2.shape[0])]
        return_foreground_point.append(point_list1)
        return_background_point.append(point_list2)
   
        
    return return_bbox, return_foreground_point, return_background_point


def modify_origin_prompts(bboxs_list,foreground_list,background_list):
    return_bboxs,return_foreground,return_background = [],[],[]
    for bboxs,foreground_points,background_points in zip(bboxs_list,foreground_list,background_list):
        return_bboxs.append([bboxs[i:i+1] for i in range(bboxs.shape[0])])
        if foreground_points is not None:
            return_foreground.append([foreground_points[i:i+1] for i in range(foreground_points.shape[0])])
        else:
            return_foreground.append(None) 
        if background_points is not None:
            return_background.append([background_points[i:i+1] for i in range(background_points.shape[0])])
        else:
            return_background.append(None)
    return return_bboxs,return_foreground,return_background









def modify_for_separated_input(input_labels, bboxs_list, foreground_points_list, background_points_list):
    bbox1, points1, points2 = modify_separated_prompts(bboxs_list, foreground_points_list, background_points_list)
    return modify_label(input_labels), bbox1, points1, points2

def modify_for_origin_input(input_labels, bboxs_list, foreground_points_list, background_points_list):
    bbox,points1,points2 = modify_origin_prompts(bboxs_list,foreground_points_list,background_points_list)
    return modify_label(input_labels),bbox,points1,points2