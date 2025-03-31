import torch
from torch.utils.tensorboard import SummaryWriter
from utils.finetune_sam.train_function import train_function,train,Draw,otherDraw
from model.decoder.mask_decoder.mask_decoderHQ import build_module
from model.build_sam import build_sam
from segmentation_module import make_model
import argparser
import shlex
import copy
import random
random.seed(42)
torch.random.manual_seed(42)
import numpy as np
np.random.seed(42)
# sam = build_sam()
# print(sam)
# sam = sam.cuda()
# sam.load_state_dict(torch.load(r"./sam_hq_vit_h.pth"))
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# model = build_module("mask_decoder_h")
# image_encoder = sam.image_encoder
# image_encoder.load_state_dict(torch.load(r"./encoder.pth"),strict=False)
# model = sam.mask_decoder
# state_dict = model.state_dict()
# torch.save(state_dict,r"./mask_decoder_vit_h.pth")
# model = build_module("mask_decoder_h")
# model.load_state_dict(torch.load(r"./mask_decoder_vit_h.pth"))
# model = copy.deepcopy(sam.mask_decoder)
# state_dict = model.state_dict()
# torch.save(state_dict,"./mask_decoder_vit_h.pth")
# print("mask decoder finish")
from utils.create_dataset import load_dataset_from_config
from utils.load_config.load_dataset_config import get_dataset_config
from utils.create_optimizer import build_optimizer

from dataset.dataloader import Dataloader

# dataloader = Dataloader(train,batch_size=4)

# for i in dataloader:
#     image = i["image"].cuda()
#     output = image_encoder(image)
#     print("finish")
import os
import torch.distributed as dist

def setup(rank,world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend="nccl",rank=rank,world_size=world_size)
    torch.cuda.set_device(rank)    



from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# sampler = DistributedSampler(train,num_replicas=dist.get_world_size(),rank=dist.get_rank())

# for i in range(3):
#     sampler.set_epoch(i)
#     for batch in dataloader:
#         print("True")
if __name__ == "__main__":
    config = get_dataset_config("VOC")
    print("finish")
    train1, val = load_dataset_from_config(config, 2, None)

    train1.update_stage(5)
    val.update_stage(5)
    dataloader = Dataloader(train1, batch_size=1)
    val_dataloader = Dataloader(val, batch_size=1)
    General_OPTIONS = "--backbone resnet101 --data_root F:\\Code_Field\\Python_Code\\Pycharm_Code\\test\\data\\PascalVOC12 --overlap --dataset voc --name test --task 10-1 --method FT --step 10"
    PLOPNeST_OPTIONS= ("--warm_up --warm_epochs 5 --warm_lr_scale 1 --unce_in_warm "
                       "--checkpoint checkpoints/plop_nest_step/ --fix_pre_cls --pod local --pod_factor 0.01 --pod_logits  "
                       "--pseudo entropy --threshold 0.001 --classif_adaptive_factor  --init_balanced")
    PLOPNeST_POD = '--pod_options \'{"switch": {"after": {"extra_channels": "sum", "factor": 0.0005, "type": "local"}}}\''
    MiB_OPTIONS = "--warm_up --warm_epochs 5 --warm_lr_scale 1 --unce_in_warm --loss_kd 10 --unce --unkd --init_balanced"
    opt_plop_nest = General_OPTIONS+" "+PLOPNeST_OPTIONS+" "+PLOPNeST_POD
    opt_mib = General_OPTIONS+" "+MiB_OPTIONS

    args_plop_nest = shlex.split(opt_plop_nest)
    parser = argparser.get_argparser()
    opts_plop_nest = parser.parse_args(args_plop_nest)
    args_mib = shlex.split(opt_mib)
    parser = argparser.get_argparser()
    opts_mib = parser.parse_args(args_mib)
    opts = [opts_plop_nest, opts_mib]
    save_path = ""
    i=0
    # model = make_model(opts,[11,1,1,1,1,1,1,1,1,1,1])
    # state_dict = torch.load("checkpoints/step/10-1-voc_ours_10.pth")["model_state"]
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k,v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # test_state_dict = model.state_dict()
    # model.load_state_dict(new_state_dict,strict=False)
    # model.cuda()
    # model.eval()
    # optim_config= {"type":"AdamW"}
    # model = model.cuda()
    # optim = build_optimizer(model,optim_config)
    # sam = sam.cuda()
    # model = model.train()
    # sam = sam.eval()
    # writer = SummaryWriter("./logs")
    for opt in opts:
        if i == 0:
            state_path = "./checkpoints/PLOP_NeST/15-5s-voc_PLOP_NeST_5.pth"
        else:
            state_path = "checkpoints/plop_step/15-5s-voc_PLOP_5.pth"
        state_dict = torch.load(state_path)["model_state"]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model = make_model(opt,[16]+[1 for j in range(5)])
        model.cuda()

        model.load_state_dict(new_state_dict, strict=False)
        if True:
            model.init_new_classifier_simplified("cuda")
        model.eval()
        val_dataloader = Dataloader(val, batch_size=1)
        otherDraw(model,val_dataloader,i)