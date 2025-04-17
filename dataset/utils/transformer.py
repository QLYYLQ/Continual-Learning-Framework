from functools import wraps
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
from PIL import Image

_change_type = [Image.Image, np.ndarray]
_target_type = [torch.Tensor]


def change_input_to_tensor(
    func: Callable[[Any], Any],
    changed_type: Optional[Iterable[type[Any]]] = None,
    target_type: Optional[Iterable[type[Any]]] = None,
):
    if changed_type is None:
        changed_type = _change_type
    if target_type is None:
        target_type = _target_type

    @wraps(func)
    def wrapper(*args, **kwargs):
        input_data = args[0]
        if isinstance(input_data, dict):
            for key in input_data.keys():
                input_data[key] = torch.from_numpy(input_data[key]).float()
        else:
            input_data = torch.from_numpy(input_data).float()
        return wrapper(input_data, *args[1:], **kwargs)
