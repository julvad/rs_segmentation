import numpy as np
import os
import torchvision.transforms as transforms
from torchvision.transforms import v2

from models.dinov3 import SegmentationModelDinoV3
from train import train_model
from load_data import get_train_val_dataloaders
from utils.transforms import basic_transform_3bands, imagenet_transform
import torch
import random

seed = 24
random.seed(seed)          
np.random.seed(seed)        
torch.manual_seed(seed)       
torch.cuda.manual_seed(seed)

PS = 20
TS_PATH = f'data/pytorch/a1_512_{PS}_pruned/global/train/'
# MODEL = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
MODEL = 'facebook/dinov3-vitl16-pretrain-sat493m'

#__________________________________________________
vit_size = MODEL.split('dinov3-')[1].split('-pretrain')[0]
if __name__ == "__main__":
    device = "cuda:0"
    lr = 1e-3
    num_epochs = 100
    batch_size = 96
    tile_size = 512
    out_log_dir = f"runs/dinov3/512_pruned/{vit_size}_{lr}_epochs_{num_epochs}_bs{batch_size}_CELoss"
    print('out log_dir:',os.path.abspath(out_log_dir))

    dinov3_model = SegmentationModelDinoV3(
        num_classes=2,
        freeze_backbone=True,
        size_output=(tile_size, tile_size),
        n_vit_feature_layers=4,
        device=device,
        head_fuse_dropout=0.1,
        model_name=MODEL,
        is_model_cached=False # Set to true if model is cached in a1/hf_cache/
    )

    transform = basic_transform_3bands(resize_size=tile_size)
    # transform = imagenet_transform(resize_size=tile_size)
    # data_aug = random_rot_flip(rot180_prob=0.3,hflip_prob=0.15, vflip_prob=0.15)
    """
    Transformations for DINOv3 model. 
    https://github.com/facebookresearch/dinov3?tab=readme-ov-file#image-transforms
    """

    train_loader, val_loader = get_train_val_dataloaders(
        path_data_dir=TS_PATH,
        batch_size=batch_size, #b2: bs 32: 24/24gb (26+?) bs 16: 16(20)/24 ##b4: bs16: 37gb bs8:19gb
        num_workers=0,
        frac_train=0.9,
        transform_train=transform,
        transform_val=transform
    )

    # imgs, masks = next(iter(train_loader))
    # res = dinov3_model(imgs.to(device))

    train_model(
        model=dinov3_model,
        train_loader=train_loader,
        val_loader=val_loader,
        log_dir=out_log_dir,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=1e-4,
    )

# tensorboard --logdir=C:\Users\juvad3723\.1\--Projects\Local\seg_test\runs\segformer\512_pruned\vit_0.0001_epochs_160_bs32_WeightedComboLoss --reload_interval 5
