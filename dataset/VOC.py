import os

import torch.utils.data as data
from .register import register_training_dataset, register_validation_dataset
from dataset.base_dataset import BaseSplit,BaseIncrement,BaseEvaluate




@register_training_dataset
class Split(BaseSplit):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)


    def _get_path(self):
        # train 和 train_aug 文件一样
        if self.train:
            return os.path.join(self.root, "splits","train.txt")
        else:
            return os.path.join(self.root, "splits","val.txt")



@register_training_dataset
class Increment(BaseIncrement):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)




@register_validation_dataset
class Val(BaseEvaluate):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)



