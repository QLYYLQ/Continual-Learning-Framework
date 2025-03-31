import torch
import numpy as np
from typing import Dict,List
from utils.finetune_sam.utils.create_finetune_sam_point_and_bbox import (find_separated_boundaries,
                                                                   create_bbox_from_origin_and_separated_label,
                                                                   create_point_from_origin_and_separated_label,
                                                                   create_point_label_from_origin_and_separated_points
                                                                   )
from utils.finetune_sam.utils.create_finetune_sam_input import modify_for_separated_input,modify_for_origin_input

def create_sam_input_dict(image:torch.Tensor,label:torch.Tensor,label_index:List,foreground_points_number=1,background_points_number=1):
    _,separated_label_list = find_separated_boundaries(label,label_index)
    origin_bbox, separated_bbox = create_bbox_from_origin_and_separated_label(label,separated_label_list)
    origin_foreground_points,origin_background_points,separated_foreground_points,separated_background_points =create_point_from_origin_and_separated_label(label,separated_label_list,"number",foreground_number=foreground_points_number,background_number=background_points_number)
    # points_label11,points_label12,points_label_21,points_label22 = create_point_label_from_origin_and_separated_points(origin_foreground_points,separated_foreground_points)
    separated_label1,separated_bbox1,separated_foreground_points1,separated_background_points1 = modify_for_separated_input(separated_label_list,separated_bbox,separated_foreground_points,separated_background_points)
    origin_label1,origin_bbox1,origin_foreground_points1,origin_background_points1 = modify_for_origin_input(label.numpy(),origin_bbox,origin_foreground_points,origin_background_points)
    return_dict1 = create_dict(image,origin_label1,origin_bbox1,origin_foreground_points1,origin_background_points1)
    return_dict2 = create_dict(image,separated_label1,separated_bbox1,separated_foreground_points1,separated_background_points1)
    separated_label1 = [a.squeeze().astype(np.float32) for a in separated_label1]
    return {"origin_label":origin_label1,"separated_label":separated_label1,"origin_dict":return_dict1,"separated_dict":return_dict2}

           

def create_dict(image:torch.Tensor,labels:np.ndarray,bboxs:List[np.ndarray],foreground_points:List[np.ndarray],background_points:List[np.ndarray]):
    batch_input = []
    for batch in range(image.shape[0]):
        single_input = []
        for bbox,label, points1,points2 in zip(bboxs[batch],labels[batch],foreground_points[batch],background_points[batch]):
            batch_dict = dict()
            batch_dict["image"] = image[batch]
            batch_dict["boxes"] = torch.as_tensor(bbox).to(image.device)
            if points1 is not None and points2 is not None:
                batch_dict["point_coords"] = torch.as_tensor(np.concatenate((points1,points2),axis=1)).to(image.device)
                postive_points_label = torch.ones(points1.shape[0:2]).to(image.device)
                negtive_points_label = torch.zeros(points2.shape[0:2]).to(image.device)
                batch_dict["point_labels"] = torch.concat((postive_points_label,negtive_points_label),dim=1).to(image.device)
            elif points1 is None and points2 is not None:
                batch_dict["point_coords"] = torch.as_tensor(points2).to(image.device)
                # postive_points_label = torch.ones(points1.shape[0:2]).to(image.device)
                negtive_points_label = torch.zeros(points2.shape[0:2]).to(image.device)
                batch_dict["point_labels"] = negtive_points_label.to(image.device)
            elif points2 is None and points1 is not None:
                batch_dict["point_coords"] = torch.as_tensor(points1).to(image.device)
                postive_points_label = torch.ones(points1.shape[0:2]).to(image.device)
                # negtive_points_label = torch.zeros(points2.shape[0:2]).to(image.device)
                batch_dict["point_labels"] = postive_points_label.to(image.device)
            # postive_points_label = torch.ones(points1.shape[0:2]).to(image.device)
            # negtive_points_label = torch.zeros(points2.shape[0:2]).to(image.device)
            # batch_dict["point_labels"] = torch.concat((postive_points_label,negtive_points_label),dim=1).to(image.device)
            batch_dict["original_size"] = image.shape[-2:]
            batch_dict["label"] = label
            single_input.append(batch_dict)
        batch_input.append(single_input)
    return batch_input
