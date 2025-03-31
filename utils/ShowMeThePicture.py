import os
from torch.nn import functional as F
from utils.finetune_sam.utils.show_image import show_picture_with_return_dict,create_color_list
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.joinpath("train_data")

def show_image_to_writer(writer,input_dict,masks_hq,file_name):
    if writer == None:
        pass


def show_many_image(input_dict,masks_hq,dir_name):
    color_list = create_color_list(len(input_dict))
    file_path = root_path.joinpath(dir_name)
    Path(file_path).mkdir(exist_ok=True)
    show_picture_with_return_dict(input_dict,str(file_path.joinpath("1")),color_list)
    for a in input_dict:
        a["point_coords"] = None
    show_picture_with_return_dict(input_dict,file_path.joinpath("2"),color_list)
    for a in input_dict:
        a["boxes"] = None
    show_picture_with_return_dict(input_dict,file_path.joinpath("3"),color_list)
    for index,a in enumerate(input_dict):
        mask = masks_hq[0,index].unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask,(1024,1024),mode="bilinear",align_corners=False)
        mask = mask>0.0
        mask = mask.detach().cpu().numpy()
        a["label"] = mask
    show_picture_with_return_dict(input_dict,file_path.joinpath("4"),color_list)
    for a in input_dict:
        a["label"] = None
    show_picture_with_return_dict(input_dict,"5",color_list)

if __name__ == "__main__":
    print(root_path)