import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
import os
import torch.multiprocessing as mp
from model.build_sam import build_sam

def ddp_set(rank,world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl",rank=rank,world_size=world_size)

# sam = build_sam(checkpoint=r"./sam_vit_h_4b8939.pth")
# print(sam)
print("start_ddp")
ddp_set(0,1)
world_size = torch.cuda.device_count()
mp.spawn()
destroy_process_group()
