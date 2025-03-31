import torch
from utils.load_config.load_dataset_config import get_dataset_config
from utils.create_dataset import load_dataset_from_config
from model.base_model import BaseModel
from model.encoder.image_encoder.swin import build_swin
from model.encoder.position_embedding.position_embedding import PositionEmbeddingSineHW
from model.query_module.transformer import Transformer
from utils.init_from_config import auto_init
from model.decoder.mask_decoder.mask_decoderHQ import MaskDecoder
from model.decoder.mask_decoder.TwoWayTransformer import TwoWayTransformer
from model.encoder.text_encoder.clip_text import clip_encoder
from dataset.dataloader import Dataloader
import torch.nn.functional as F
config = get_dataset_config("VOC")
dataset = load_dataset_from_config(config,2,None)

# model = BaseModel()
print("start")
image_encoder = build_swin("swin_B_384_22k")

# print(image_encoder)
image_encoder.load_state_dict(torch.load(r"./encoder.pth"),strict=False)
print("finish")

positon_embedding = PositionEmbeddingSineHW(128,temperatureH=20,temperatureW=20,normalize=True)
positon_embedding.load_state_dict(torch.load(r"./position_embedding.pth"),strict=False)
print("finish")

mask_transformer = TwoWayTransformer( depth=2,embedding_dim=256,mlp_dim=2048,num_heads=8)
mask_decoder = MaskDecoder(transformer=mask_transformer,num_multimask_outputs=3,transformer_dim=256,iou_head_depth=3,iou_head_hidden_dim=256,encoder_embedding_dim=1280)
mask_decoder.load_state_dict(torch.load(r"./mask_decoder.pth"),strict=False)
print("finish")

text_encoder = clip_encoder()
print("finish")



num_channels = [256,512,1024]
num_backbone_outs = 3
from torch import nn
input_proj_list = []
hidden_dim = 256
num_feature_levels=4
for _ in range(num_backbone_outs):
    in_channels = num_channels[_]
    input_proj_list.append(
        nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        )
    )
for _ in range(num_feature_levels - num_backbone_outs):
    input_proj_list.append(
        nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, hidden_dim),
        )
    )
    in_channels = hidden_dim
input_proj = nn.ModuleList(input_proj_list)
input_proj.load_state_dict(torch.load(r"./none.pth"),strict=False)
print("finish")




query_net = Transformer(
    d_model=256,
        dropout=0.0,
        nhead=8,
        num_queries=900,
        dim_feedforward=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        normalize_before=False,
        return_intermediate_dec=True,
        query_dim=4,
        activation='relu',
        num_patterns=0,
        num_feature_levels=4,
        enc_n_points=4,
        dec_n_points=4,
        learnable_tgt_init=True,
        # two stage
        two_stage_type="standard",  # ['no', 'standard', 'early']
        embed_init_tgt=True,
        use_text_enhancer=True,
        use_fusion_layer=True,
        use_checkpoint=True,
        use_transformer_ckpt=True,
        use_text_cross_attention=True,
        text_dropout=0.0,
        fusion_dropout=0.0,
        fusion_droppath=0.0,
)















query_net.load_state_dict(torch.load(r"./transformer.pth"),strict=False)
print("finish")













dataloader = Dataloader(dataset, 2)
for image,masks,texts in dataloader:
    text_embedding = []
    for text in texts:
        text_out = text_encoder(text)
        text_embedding.append(text_out)
    output = image_encoder(image,masks)
    pos = []
    for k, (img,mask) in output.items():
        po = positon_embedding(img,mask)
        pos.append(po)
    out = input_proj[3](output[2][0])
    out_img = [input_proj[k](a[0])for k,a in output.items()]+[out]
    mask = F.interpolate(masks[None].float(),out.shape[-2]).to(torch.bool).squeeze()
    mask_img = [a[1] for k,a in output.items()]+[mask]
    pos1 = positon_embedding(out,mask)
    pos+=[pos1]
    print(output)
    # images = [a.tensor for a in output]
    for img,text in zip(output,text_embedding):
        for text1 in text:
            text1["encoded_text"] = text1["encoded_text"].unsqueeze(0)
            query = query_net(out_img, mask_img, None, pos, None, None, text1)
