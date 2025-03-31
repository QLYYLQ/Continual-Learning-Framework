import torch
import torch.distributed
import random
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from dataset.dataloader import Dataloader
from utils.finetune_sam.utils.misc import MetricLogger
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils.create_optimizer import build_optimizer
from utils.finetune_sam.train_function import RunAsBatch_stage1
from utils.finetune_sam.train_function import Loss
import os
from utils.create_dataset import load_dataset_from_config
from utils.load_config.load_dataset_config import get_dataset_config
from model.decoder.mask_decoder.mask_decoderHQ import build_module
from model.build_sam import build_sam
from collections import defaultdict
from typing import Dict
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class SAMTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        sam: torch.nn.Module,
        train_data: Dataset,
        val_data: Dataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        eval_every: int,
        batch_size:int,
        classes:Dict=None,
        seed:int=42
    ) -> None:
        self.gpu_id = gpu_id
        self.classes = classes
        self.model = model.to(gpu_id)
        self.sam = sam.to(gpu_id)
        self.train_data = self.prepare_dataloader(train_data,batch_size)
        self.val_data = self.prepare_dataloader(val_data,batch_size)
        self.optimizer = optimizer
        self.save_every = save_every
        self.sam = DDP(sam,device_ids=[gpu_id])
        self.model = DDP(model, device_ids=[gpu_id],find_unused_parameters=True)
        self.metric_logger = MetricLogger(delimiter="  ")
        self.loss = Loss()
        self.eval_every = eval_every
        self.set_seed_for_ddp(seed)
        if self.gpu_id == 0:
            self.writer = SummaryWriter("/root/tf-logs")
    def prepare_dataloader(self,dataset: Dataset, batch_size: int):
        return Dataloader(
            dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(dataset),
            worker_init_fn=lambda worker_id: np.random.seed(self.seed+self.gpu_id)
        )

    def set_seed_for_ddp(self,seed:int):
        seed = seed+self.gpu_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        print(f"[GPU{self.gpu_id}] Seed {seed}")

    def _run_batch(self, batch,step):
        self.optimizer.zero_grad()
        return_list = RunAsBatch_stage1(self.model,self.sam,batch,self.optimizer,self.loss)
        # output = self.model(source)
        # loss = F.cross_entropy(output, targets)
        # loss.backward()
        # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        print(return_list[0])
        if step % 200 == 0:
            if self.gpu_id == 0:
                with torch.no_grad():
                    self.writer.add_scalar("total_loss",sum(return_list[0])/len(return_list[0]),step)
                    ce_loss ,dice_loss= 0,0
                    for loss_dict in return_list[1]:
                        for name,loss in loss_dict.items():
                            if name == "loss_dice":
                                dice_loss += loss.item()
                            if name == "loss_ce":
                                ce_loss += loss.item()
                    ce_loss = ce_loss/len(return_list[1])
                    dice_loss = dice_loss/len(return_list[1])

                    self.writer.add_scalar("ce_loss",ce_loss,step)
                    self.writer.add_scalar("dice_loss",dice_loss,step)
        # self.optimizer.step()
        torch.distributed.barrier()

    def _run_eval_batch(self,batch,step):
        pass


    def _run_epoch(self, epoch,train_step,eval_step):
        # total_len = len(self.train_data)
        
        b_sz = len(next(iter(self.train_data)))
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if epoch % self.eval_every == 0:
            self.val_data.sampler.set_epoch(epoch)
            # for batch in enumerate(self.val_data):
            #     self._run_eval_batch(batch,eval_step)
            #     eval_step+=1

        self.train_data.sampler.set_epoch(epoch)
        for index,batch in enumerate(self.train_data):
            # batch = batch.to(self.gpu_id)
            # targets = targets.to(self.gpu_id)
            self._run_batch(batch,train_step)
            train_step +=1
        return train_step,eval_step


    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"/root/autodl-tmp/test/saved_checkpoint/fintune_sam/{epoch}.pth"
        torch.save(ckp, PATH)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        train_step = 0
        eval_step = 0
        for epoch in range(max_epochs):
            train_step,eval_step = self._run_epoch(epoch,train_step,eval_step)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    # train_set = MyTrainDataset(2048)  # load your dataset
    config = get_dataset_config("ADE")
    dataset,val_dataset = load_dataset_from_config(config,3,None)
    model = build_module("mask_decoder_h")  # load your model
    model.load_state_dict(torch.load(r"./mask_decoder_vit_h.pth"))
    sam = build_sam()
    sam.load_state_dict(torch.load(r"./sam_hq_vit_h.pth"))
    optim_config = {"type":"AdamW"}
    optimizer = build_optimizer(model,optim_config)
    return dataset,val_dataset, model,sam, optimizer,dataset.stage_class




def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int,eval_every:int):
    ddp_setup(rank, world_size)
    dataset,val_dataset, model,sam, optimizer,classes = load_train_objs()

    trainer = SAMTrainer(model,sam, dataset, val_dataset,optimizer, rank, save_every,eval_every,batch_size,classes)
    trainer.train(total_epochs)
    destroy_process_group()


def start_train():
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=4, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default = 1,type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--eval_every', default=1, type=int, help='How often to save a snapshot')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, args.eval_every), nprocs=world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)