import torch
from torch.utils.data import DataLoader as PytorchDataLoader
from utils.ImageList import ImageList
from utils.finetune_sam.utils.create_finetune_sam_point_and_bbox import (find_separated_boundaries,
                                                                   create_bbox_from_origin_and_separated_label,
                                                                   create_point_from_origin_and_separated_label,
                                                                   create_point_label_from_origin_and_separated_points
                                                                   )
from utils.finetune_sam.utils.create_finetune_sam_input import modify_for_separated_input,modify_for_origin_input
class DataLoader(PytorchDataLoader):
    def __init__(self, dataset, batch_size=1,foreground_points_number=5,background_points_number = 5,**kwargs):
        collate_fn = self.collate_fn
        super(DataLoader, self).__init__(dataset, batch_size, collate_fn=collate_fn,**kwargs)
        self.foreground_points_number = foreground_points_number
        self.backgournd_points_number = background_points_number

    @staticmethod
    def collate_fn(batch):
        """这里实现sam中需要的输入的一切逻辑"""
        data = [item["data"][0].tensor for item in batch]
        mask = [item["data"][0].mask for item in batch]
        label = [item["data"][1].tensor for item in batch]
        label_index = [item["label_index"] for item in batch]
        text_prompt = [item["text_prompt"] for item in batch]
        image_size = [item["data"][0].image_sizes for item in batch]
        data = torch.stack(data)
        label = torch.stack(label)
        mask = torch.stack(mask)
        image_dict = {"image": ImageList(data, mask, image_size), "target": ImageList(label, mask, image_size)}
        return {"image": data, "label": label, "label_index":label_index,"mask": mask, "text-prompt": text_prompt, "image_dict": image_dict}

    def _create_sam_input(self,image,label,label_index):
        _,separated_labels_list = find_separated_boundaries(label,label_index)
        origin_bbox,separated_bbox = create_bbox_from_origin_and_separated_label(label,separated_labels_list)
        origin_foreground_points,origin_background_points,separated_foreground_points,separated_background_points =create_point_from_origin_and_separated_label(label,separated_labels_list,"number",foreground_number=self.foreground_points_number,background_number=self.backgournd_points_number)
        points_label11,points_label12,points_label_21,points_label22 = create_point_label_from_origin_and_separated_points(origin_foreground_points,separated_foreground_points)
