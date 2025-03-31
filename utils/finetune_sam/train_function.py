import numpy as np
import torch
import torch.distributed
from torch.nn import functional as F
from tqdm import tqdm
from utils.finetune_sam.utils.from_label_to_sam_input import create_sam_input_dict
from .utils.show_image import show_picture_with_return_dict, create_color_list, hex_to_rgba, voc_cmap
from utils.finetune_sam.utils.misc import reduce_dict
from utils.loss.dice import MultiClassDiceLoss
from utils.compute_iou import calculate_iou
from utils.Tensor_boardWriter import write_iou_label
from utils.ShowMeThePicture import show_many_image
from utils.create_optimizer import build_optimizer
from collections import defaultdict


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.dice = MultiClassDiceLoss()
        self.ce = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        assert pred.size() == target.size()
        dice_loss = self.dice(pred, target)
        ce_loss = self.ce(pred, target)
        return dice_loss, ce_loss


def train_function(model, train_loader, sam, optim, epoch, writer):
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()

    # optim = build_optimizer(model, optimizer_config)
    stage = 1

    for i in range(epoch):
        for batch in train_loader:
            image = batch['image'].cuda()
            label = batch['label']
            mask = batch['mask']
            label_index = batch['label_index']
            return_dict = create_sam_input_dict(image, label, label_index, foreground_points_number=0,
                                                background_points_number=0)

            return_list = return_dict['origin_dict']

            for batch_index, batch_input in enumerate(return_list):
                # show_picture_with_return_dict(return_dict['separated_dict'][batch_index],"2")
                show_picture_with_return_dict(return_list[batch_index], '1')
                with torch.no_grad():
                    batched_output, interim_embeddings = sam(batch_input, multimask_output=False)

                for index, output in enumerate(batched_output):
                    batch_input[index]["point_coords"] = None
                    batch_input[index]["boxes"] = None
                show_picture_with_return_dict(batch_input, "3")
                batch_len = len(batched_output)
                encoder_embedding = batched_output[0]['encoder_embedding']
                image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
                sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
                dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

                mask_hq = model(image_embeddings=encoder_embedding,
                                image_pe=image_pe,
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                                hq_token_only=False,
                                interm_embeddings=interim_embeddings)
                masks_hq = mask_hq[0].permute(1, 0, 2, 3)
                for index, output in enumerate(batched_output):
                    mask = masks_hq[0, index].unsqueeze(0).unsqueeze(0)
                    mask = F.interpolate(mask, (1024, 1024), mode="bilinear", align_corners=False)
                    mask = mask > 0.0
                    mask = mask.cpu().detach().numpy()
                    # batch_input[index]["boxes"] = None
                    batch_input[index]["label"] = mask
                show_picture_with_return_dict(batch_input, "4")
                for index, output in enumerate(batched_output):
                    batch_input[index]["label"] = None
                    batch_input[index]["boxes"] = None
                show_picture_with_return_dict(batch_input, "6")
                masks_hq = mask_hq[0].permute(1, 0, 2, 3)
                masks_hq = F.interpolate(masks_hq, (1024, 1024))
                masks_hq = masks_hq > 0.0

                size = batch["image_dict"]["image"].image_sizes[batch_index]
                masks_hq = masks_hq[:, :, :size[0], :size[1]]
                label = return_dict["origin_label"][batch_index][:, :size[0], :size[1]]
                label = torch.tensor(label)[None]
                # loss_value = loss(masks_hq,label)
                iou_list, iou = calculate_iou(label.cuda(), masks_hq.cuda())
                print(iou_list)
                pass
            if stage == 1:
                continue
            elif stage == 2:
                continue

    return None


def eval_function(model, eval_loader, sam, optim, epoch, writer):
    model.eval()


def FastRun(sam,batch,path):
    color_list = voc_cmap(21,normalized=True)
    image = batch['image'].cuda()
    label = batch['label']
    label_index = batch['label_index']
    return_dict = create_sam_input_dict(image, label, label_index, foreground_points_number=12,
                                        background_points_number=12)

    input_prompt = return_dict["origin_dict"]
    with torch.no_grad():
        for batch_index, (batch_input,label_list) in enumerate(zip(input_prompt,label_index)):
            batched_output, interim_embeddings = sam(batch_input, multimask_output=False)
            image_size = batch["image_dict"]["image"].image_sizes[0]
            for i,j in zip(batch_input,batched_output):
                i["image"] = i["image"][...,:image_size[0],:image_size[1]]
                i["image"] = torch.zeros_like(i["image"])
                i["point_coords"] = None
                i["boxes"] = None
                i["label"] = j["masks"].squeeze()[...,:image_size[0],:image_size[1]].cpu().numpy().astype(np.uint8)
            show_picture_with_return_dict(batch_input, path, color_input=color_list, label_list=label_list,type="plate")

def RunAsBatch_stage1(model, sam, batch, optim, loss):
    image = batch['image'].cuda()
    label = batch['label']
    # mask = batch['mask']
    label_index = batch['label_index']
    return_dict = create_sam_input_dict(image, label, label_index, foreground_points_number=12,
                                        background_points_number=12)
    # show_picture_with_return_dict(return_dict["separated_dict"][3],"1")
    return_list = return_dict["origin_dict"]  # +return_dict['separated_dict']
    labels = return_dict["origin_label"]  #  + return_dict["separated_label"]
    loss_values = []
    loss_value_dicts = []
    color_list = voc_cmap(21, normalized=True)
    # color_list = create_color_list(151)
    # color_list = np.array([[245,189,218,0.7],[203,72,82,0.7]])
    # color_list = [
    #     hex_to_rgba('#CB4852', alpha=0.8),
    #     hex_to_rgba('#F5BDDA', alpha=0.8),
    # ]

    for batch_index, (batch_input, label_list) in enumerate(zip(return_list, label_index)):

        with torch.no_grad():
            batched_output, interim_embeddings = sam(batch_input, multimask_output=False)
        for i in batch_input:
            i["point_coords"] = None
            i["boxes"] = None
        show_picture_with_return_dict(batch_input, "gt", color_input=color_list, label_list=label_list)
        for i, j in zip(batch_input, batched_output):
            i["label"] = j["masks"].squeeze().cpu().numpy().astype(np.uint8)
        show_picture_with_return_dict(batch_input, "pred", color_input=color_list, label_list=label_list)
        for index, i in enumerate(batch_input):
            i["label"] = None
        show_picture_with_return_dict(batch_input, "image", color_input=color_list, label_list=label_list)
        # show_picture_with_return_dict(batch_input,"pred_2",color_input=color_list,type=1)

        batch_len = len(batched_output)
        # encoder_embedding = batched_output[0]['encoder_embedding']
        # image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
        # sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
        # dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
        # mask_hq = model(image_embeddings=encoder_embedding,
        #         image_pe=image_pe,
        #         sparse_prompt_embeddings=sparse_embeddings,
        #         dense_prompt_embeddings=dense_embeddings,
        #         multimask_output=False,
        #         hq_token_only=True,
        #         interm_embeddings=interim_embeddings)
        # masks_hq = mask_hq.permute(1,0,2,3)
        # masks_hq = F.interpolate(masks_hq,(1024,1024))
        # if batch_index >= len(return_list)/2:
        #     size = batch["image_dict"]["image"].image_sizes[int(batch_index-len(return_list)/2)]
        # else:
        #     size = batch["image_dict"]["image"].image_sizes[batch_index]
        # # masks_hq = masks_hq[:,:,:size[0],:size[1]]
        # if len(labels[batch_index].shape) == 2:
        #     labels[batch_index] = labels[batch_index][None]
        #     # print("stop")
        # label = labels[batch_index]# [:,:size[0],:size[1]]
        # label = torch.tensor(label)[None].cuda()
        # dice_loss,ce_loss = loss(masks_hq,label)
        # # masks_hq = masks_hq[:,:,:size[0],:size[1]]
        # loss_value = dice_loss + ce_loss
        # loss_dict = {"loss_dice":dice_loss,"loss_ce":ce_loss}
        # loss_dict_reduced = reduce_dict(loss_dict)
        # losses_reduced_scaled = sum(loss_dict_reduced.values())
        # loss_value_scaled = losses_reduced_scaled.item()
        # torch.distributed.barrier()
        # optim.zero_grad()
        # loss_value.backward()
        # optim.step()
        # loss_values.append(loss_value_scaled)
        # loss_value_dicts.append(loss_dict_reduced)
    # del  label
    #     # del mask_hq, masks_hq
    #     # del dice_loss, ce_loss, loss_value
    #     # del loss_dict, loss_dict_reduced, losses_reduced_scaled
    #     # del encoder_embedding, image_pe, sparse_embeddings, dense_embeddings
    # del return_dict, return_list, labels
    # del batched_output, interim_embeddings
    torch.cuda.empty_cache()
    # print(loss_value_scaled)
    return loss_values, loss_value_dicts


def Draw(model, dataloader):
    model = model.eval()
    for index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        DrawWithSingleClass(model, batch, "OtherMethod_Deeplab", index)


def otherDraw(model, dataloader,pth_number):
    model = model.eval()
    for index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if index <=100 and index in [67,84,23,2,88,99]:
            DrawForOtherMethod(model, batch, "OtherMethod", index,pth_number)
        elif index >100:
            break
        else:
            pass

def Drawpicturelabel(dataloader,images_list=[67,84,23,2,88,99]):
    color_list = voc_cmap(21, normalized=True)
    for index, batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        if index in images_list:
            image = batch['image']
            label = batch['label']
            label_list = batch['label_index']
            image_list = []
            for label_index in label_list[0]:
                image_dict = {}
                image_dict["image"] = image.squeeze()
                image_dict["label"] = None
                image_dict["boxes"] = None
                image_dict["point_coords"] = None
                image_list.append(image_dict)
            show_picture_with_return_dict(image_list,f"OtherMethod/{index}_image",color_list,label_list[0])
            for i,label_index in enumerate(label_list[0]):
                label1 = label==label_index
                image_list[i]["label"] = label1.squeeze()
            show_picture_with_return_dict(image_list,f"OtherMethod/{index}_label",color_list,label_list[0])
def train(model, train_loader, val_loader, sam, optim, epoch, writer):
    model = model.train()
    sam = sam.eval()
    loss = Loss()
    # optim = build_optimizer(model,optim_config)
    # optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))
    # for index,batch in tqdm(enumerate(val_loader),total=len(val_loader)):
    #     EvalAsBatch(model,sam,batch,index,loss,writer)
    for batch in tqdm(train_loader, total=len(train_loader)):
        text_prompt = batch["text-prompt"]

        RunAsBatch_stage1(model, sam, batch, optim, loss)
        print("finish")


def EvalAsBatch(model, sam, batch, epoch, loss, writer):
    model = model.eval()
    sam = sam.eval()
    image = batch['image'].cuda()
    label = batch['label']
    masks = batch['mask']
    label_index = batch['label_index']
    return_dict = create_sam_input_dict(image, label, label_index, foreground_points_number=7,
                                        background_points_number=7)

    return_list = return_dict["origin_dict"]

    iou_list = []
    for batch_index, batch_input in enumerate(return_list):
        mask = masks[batch_index].squeeze().cuda()
        with torch.no_grad():
            batched_output, interim_embeddings = sam(batch_input, multimask_output=False)
        batch_len = len(batched_output)
        encoder_embedding = batched_output[0]['encoder_embedding']
        image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
        sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
        dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
        with torch.no_grad():
            mask_hq = model(image_embeddings=encoder_embedding,
                            image_pe=image_pe,
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            hq_token_only=False,
                            interm_embeddings=interim_embeddings)
        masks_hq = mask_hq[0].permute(1, 0, 2, 3)
        show_many_image(batch_input, masks_hq, "test")

        masks_hq = F.interpolate(masks_hq, (1024, 1024))
        masks_hq = masks_hq > 0.0
        size = batch["image_dict"]["image"].image_sizes[batch_index]
        masks_hq = masks_hq[:, :, :size[0], :size[1]]
        label = return_dict["origin_label"][batch_index][:, :size[0], :size[1]]
        label = torch.tensor(label)[None].cuda()
        iou_list, iou_mean = calculate_iou(label, masks_hq)
        # print(iou)
        # print(loss_value)
        # write_iou_label(writer,iou_list,batch["text-prompt"][batch_index],epoch)


def DrawWithSingleClass(sam, batch, save_path, BatchNumber):
    image = batch['image'].cuda()
    label = batch['label']
    label_index = batch['label_index']
    text_prompt = batch["text-prompt"]
    image_shape = batch["image_dict"]["image"].image_sizes[0]
    # mask = batch['mask']
    label_index = batch['label_index']
    return_dict = create_sam_input_dict(image, label, label_index, foreground_points_number=12,
                                        background_points_number=12)
    # show_picture_with_return_dict(return_dict["separated_dict"][3],"1")
    return_list = return_dict["origin_dict"]
    color_list = voc_cmap(21, normalized=True)
    for batch_index, (batch_input, label_list) in enumerate(zip(return_list, label_index)):

        with torch.no_grad():
            batched_output, interim_embeddings = sam(batch_input, multimask_output=False)
        for i in batch_input:
            i["point_coords"] = None
            i["boxes"] = None
        show_picture_with_return_dict(batch_input, f"./{save_path}/{BatchNumber}_gt", color_input=color_list,
                                      label_list=label_list)
        for index, i in enumerate(batch_input):
            i["label"] = None
        show_picture_with_return_dict(batch_input, f"./{save_path}/{BatchNumber}_image", color_input=color_list,
                                      label_list=label_list)
        for i, j, label, text in zip(batch_input, batched_output, label_index[0], text_prompt[0]):
            i["label"] = j["masks"].squeeze().cpu().numpy().astype(np.uint8)
            show_picture_with_return_dict(batch_input, f"./{save_path}/{BatchNumber}_{text}_{label}_pred",
                                          color_input=color_list, label_list=label_list)

    # print(loss_value_scaled)


VOCCLASS = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'dining table',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'potted plant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tv monitor'
}


def DrawForOtherMethod(model, batch, save_path, BatchNumber, pth_number,classes=VOCCLASS):
    color_list = voc_cmap(21, normalized=True)
    image = batch['image'].cuda()
    label = batch['label']
    label_index = batch['label_index']
    text_prompt = batch["text-prompt"]
    image_shape = batch["image_dict"]["image"].image_sizes[0]
    # mask = batch['mask']
    label_index = batch['label_index'][0]
    outputs = model(image)#, ret_intermediate=False)
    if isinstance(outputs,tuple):
        outputs = outputs[0]
    _, prediction = outputs.max(dim=1)
    return_dict = []
    pred_labels = torch.unique(prediction).cpu().numpy()
    pred_labels = pred_labels[pred_labels != 0]
    for pred_label in pred_labels:
        image_dict = {}
        image_dict["image"] = image.squeeze()
        label1 = prediction == pred_label
        image_dict["label"] = label1.squeeze().cpu()
        image_dict["point_coords"] = None
        image_dict["boxes"] = None
        return_dict.append(image_dict)
        # show_picture_with_return_dict(return_dict, f"./{save_path}/{BatchNumber}_{classes[pred_label]}_pred",
        #                               color_input=color_list,
        #                               label_list=pred_labels)
    if return_dict:
        show_picture_with_return_dict(return_dict, f"./{save_path}/{BatchNumber}_{pth_number}_pred", color_input=color_list,
                                      label_list=pred_labels)
    return_list = []
    # for label_number in label_index:
    #     image_dict = {}
    #     label1 = label == label_number
    #     image_dict["label"] = label1.squeeze().cpu()
    #     image_dict["point_coords"] = None
    #     image_dict["boxes"] = None
    #     image_dict["image"] = image.squeeze()
    #     return_list.append(image_dict)
    # show_picture_with_return_dict(return_list,f"./{save_path}/{pth_number}_{BatchNumber}_image", color_input=color_list,label_list=label_index)
