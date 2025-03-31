import torch
from torch import nn
from model.build_sam import build_sam

class RunWithSam(nn.Module):
    def __init__(self):
        super(self,RunWithSam).__init__()
        self.sam = build_sam(checkpoint=r"/root/autodl-tmp/test/sam_ha_vit_h.pth")
        self.model = load_model