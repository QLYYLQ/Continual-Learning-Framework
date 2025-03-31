from utils import load_dataset_from_config
from utils import get_dataset_config
from dataset import Dataloader


config = get_dataset_config("ADE")
train_dataset,val_dataset = load_dataset_from_config(config,2,None)

train_loader = Dataloader(train_dataset,5)
val_loader = Dataloader(val_dataset,5)

epoch = 10

