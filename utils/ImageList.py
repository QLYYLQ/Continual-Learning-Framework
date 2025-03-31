from typing import Any, List, Tuple, Optional
import torch
from torch import device
from torch.nn import functional as F


def shapes_to_tensor(x: Tuple, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.

    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all(
            [isinstance(t, torch.Tensor) for t in x]
        ), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)


class ImageList(object):
    """
    这个类可以存储一个batch的tensor，并且通过__getitem__方法可以获得它们本身的大小
    例如我把一个大小是720*720的图片和一个大小是1080*720的图片放在一个batch里面，理所应当的是我要填充小图片，我们通过from_tensor函数就
    可以把它们组成一个batch，我们通过__getitem__获得图像的原始大小，通过.tensor属性直接获得这个batch的全部数据
    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w).
            During tracing, it becomes list[Tensor] instead.
    """

    def __init__(self, tensor: torch.Tensor, mask: torch.Tensor, image_sizes: Optional[List[Tuple[int, int]]] = None):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
        self.tensor = tensor
        self.mask = mask
        if mask.shape[-2:] != tensor.shape[-2:]:
            raise ValueError(
                f"ImageList.mask have the wrong shape!, which is {mask.shape} but class hopes {tensor.shape}")
        self.image_sizes = image_sizes if image_sizes is not None else None

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        if self.image_sizes is not None:
            size = self.image_sizes[idx]
            return self.tensor[idx, ..., : size[0], : size[1]]
        else:
            print("not origin image")
            return self.tensor[idx]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs)
        return ImageList(cast_tensor, cast_mask, self.image_sizes)

    @property
    def device(self) -> device:
        return self.tensor.device


def from_tensors(
        tensors: List[torch.Tensor], size_divisibility: int = 0, pad_value: float = 0.0
) -> "ImageList":
    """
    Args:
        tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
            (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
            to the same shape with `pad_value`.
        size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
            the common height and width is divisible by `size_divisibility`.
            This depends on the model and many models need a divisibility of 32.
        pad_value (float): value to pad

    Returns:
        an `ImageList`.
    """
    assert len(tensors) > 0
    assert isinstance(tensors, (tuple, list))
    for t in tensors:
        assert isinstance(t, torch.Tensor), type(t)
        assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

    image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
    image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
    max_size = torch.stack(image_sizes_tensor).max(0).values

    if size_divisibility > 1:
        stride = size_divisibility
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

    # handle weirdness of scripting and tracing ...
    if torch.jit.is_scripting():
        max_size: torch.Tensor = max_size.to(dtype=torch.long).tolist()
    else:
        if torch.jit.is_tracing():
            image_sizes = image_sizes_tensor

    if len(tensors) == 1:
        # This seems slightly (2%) faster.
        # check whether it's faster for multiple images as well
        image_size = image_sizes[0]
        padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
        batched_images = F.pad(tensors[0], padding_size, value=pad_value, ).unsqueeze_(0).to(device=tensors[0].device)
        batched_masks = torch.ones([1, 1, max_size[-2], max_size[-1]], dtype=torch.bool, device=tensors[0].device)
        batched_masks[:, :, :image_size[0], :image_size[1]] = False
    else:
        # max_size can be a tensor in tracing mode, therefore convert to list
        batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
        batched_images = tensors[0].new_full(batch_shape, pad_value, device=tensors[0].device)
        batched_masks = torch.ones(batch_shape, dtype=torch.bool, device=tensors[0].device)
        # img 原图， pad_img 是填充好的图
        for img, pad_img, mask in zip(tensors, batched_images, batched_masks):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
            mask[..., : img.shape[-2], : img.shape[-1]] = False

    return ImageList(batched_images.contiguous(), batched_masks.contiguous(), image_sizes)


if __name__ == "__main__":
    a = torch.tensor([[1, 2, 3], [1, 2, 3]])
    b = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    tensor_list = [a, b]
    imagelist = from_tensors(tensor_list, size_divisibility=3)
