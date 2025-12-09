import numpy as np
import os
import torchvision.transforms as transforms
from torchvision.transforms import v2

from models.segformer import SegmentationModelSegformer
from train import train_model
from load_data import get_train_val_dataloaders
from utils.transforms import basic_transform_3bands, random_rot_flip
import torch
import random

seed = 24
random.seed(seed)          
np.random.seed(seed)        
torch.manual_seed(seed)       
torch.cuda.manual_seed(seed)

#__________________________________________________
if __name__ == "__main__":
    device = "cuda:0"
    lr = 1e-3
    num_epochs = 160
    batch_size = 16
    tile_size = 512
    out_log_dir = f"runs/segformer_b3/512_pruned/vit_{lr}_epochs_{num_epochs}_bs{batch_size}_CELoss"
    print('out log_dir:',os.path.abspath(out_log_dir))

    segformer_model = SegmentationModelSegformer(
        num_classes=2,
        freeze_backbone=True,
        size_output=(tile_size, tile_size),
        device=device,
        head_fuse_dropout=0.1,
    )

    path_folder = f'data/512_pruned/train/'

    # transform = transforms.Compose([
    #     transforms.Lambda(lambda x: 10 *np.log1p(np.clip(x, 0, None))),
    #     transforms.Lambda(lambda x: x / (x.max() + 1e-6)),
    #     v2.Resize((512, 512), v2.InterpolationMode.BILINEAR),
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=False),
    #     transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)),
    # ])
    transform = basic_transform_3bands(resize_size=tile_size)
    data_aug = random_rot_flip(rot180_prob=0.3,hflip_prob=0.15, vflip_prob=0.15)
    """
    Transformations for DINOv3 model. 
    https://github.com/facebookresearch/dinov3?tab=readme-ov-file#image-transforms
    """

    train_loader, val_loader = get_train_val_dataloaders(
        path_data_dir=path_folder,
        batch_size=batch_size, #b2: bs 32: 24/24gb (26+?) bs 16: 16(20)/24 ##b4: bs16: 37gb bs8:19gb
        num_workers=0,
        frac_train=0.9,
        transform_train=transform,
        transform_val=transform,
        data_aug=data_aug
    )

    # imgs, masks = next(iter(train_loader))
    # res = segformer_model(imgs.to(device))

    train_model(
        model=segformer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        log_dir=out_log_dir,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=1e-4,
    )

# tensorboard --logdir=C:\Users\juvad3723\.1\--Projects\Local\seg_test\runs\segformer\512_pruned\vit_0.0001_epochs_160_bs32_WeightedComboLoss --reload_interval 5
