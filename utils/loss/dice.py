import torch
from torch import nn
from torch.nn import functional as F
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

dice_loss_jit = torch.jit.script(
    dice_loss
)

class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-5):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, predictions, targets_one_hot):
        # 预测形状: (N, C, H, W)
        # 目标形状: (N, H, W)
        N, C, H, W = predictions.shape
        
        # 将预测转换为概率
        probs = torch.sigmoid(predictions)
        
        # 将目标转换为one-hot编码
        # targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        
        # 计算每个类别的Dice系数
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        
        # 计算Dice Loss
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 应用权重（如果提供）
        if self.weight is not None:
            dice = dice * self.weight
        
        # 返回平均Dice Loss
        return 1 - dice.mean()