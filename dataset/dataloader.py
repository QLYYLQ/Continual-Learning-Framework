import torch
from torch.utils.data import DataLoader as PytorchDataLoader
from utils.ImageList import ImageList


class Dataloader(PytorchDataLoader):
    def __init__(self, dataset, batch_size=1,**kwargs):
        collate_fn = self.collate_fn
        super(Dataloader, self).__init__(dataset, batch_size, collate_fn=collate_fn,**kwargs)

    @staticmethod
    def collate_fn(batch):
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
