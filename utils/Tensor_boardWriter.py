from torch.utils.tensorboard import SummaryWriter
def write_iou_label(writer:SummaryWriter,iou_list,label_list,epoch):
    for i,(iou,label) in enumerate(zip(iou_list,label_list)):
        writer.add_scalar(f"IoU/{label}",iou,epoch)