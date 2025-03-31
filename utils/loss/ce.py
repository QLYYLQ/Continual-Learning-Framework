import torch
from torch import nn
from torch.nn import functional as F
def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)



class MaskCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(MaskCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target):
        # 确保输入是浮点型
        pred = pred.float()
        target = target.float()

        # 将预测值限制在一个小的范围内，以避免数值不稳定性
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)

        # 计算二值交叉熵损失
        bce = - (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        if self.weight is not None:
            bce = bce * self.weight.view(1, -1, 1, 1)

        if self.reduction == 'mean':
            return torch.mean(bce)
        elif self.reduction == 'sum':
            return torch.sum(bce)
        else:  # 'none'
            return bce
