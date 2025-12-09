import os
import numpy as np
import segmentation_models_pytorch as smp
from train import train_model
from load_data import get_train_val_dataloaders
from utils.transforms import basic_transform, imagenet_transform, sar_transform
import torch
import random

"""
Encoder list - https://smp.readthedocs.io/en/latest/encoders.html#
ex:
-ResNet101 
-efficientnet-b3
-ResNext
-tu-convnextv2_tiny
-CoatNext
-tu-maxvit_tiny_tf_512
-mit_b3
-PvTv2


Segmentation head list - https://smp.readthedocs.io/en/latest/models.html
ex:
-UNetPlusPlus
-DeepLabV3Plus
-UPerNet
-Segformer
"""

PS = 20 # pixel_size
TS_PATH = f'data/pytorch_big/train'
ENCODER='mit_b1'
SEG_MODEL='DeepLabV3Plus'
WEIGHTS='imagenet'
seed = 24


#___________________
random.seed(seed)          
np.random.seed(seed)        
torch.manual_seed(seed)       
torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    device = "cuda:0"
    lr = 1e-4
    dropout_rate = 0.1
    num_epochs = 100
    batch_size = 32
    tile_size = 512
    out_log_dir = f'runs/pytorch/512_{PS}/{SEG_MODEL}_{ENCODER}/lr{lr}_epochs_{num_epochs}_bs{batch_size}_CELoss'
    print('Tensorboard out log_dir:',os.path.abspath(out_log_dir))
    
    model = smp.create_model(
        arch=SEG_MODEL,                     # name of the architecture, # see args at https://smp.readthedocs.io/en/latest/models.html#deeplabv3plus
        encoder_name=ENCODER,
        encoder_weights="imagenet",
        in_channels=1,
        classes=2,
    )

    # model = smp.from_pretrained('runs/pytorch/512_20_pruned3570/UPerNet_efficientnet-b0/lr0.0001_epochs_100_bs24_CELoss/best_model_epoch_22_0.2015')
    
    # model.freeze_encoder() ## Doesnt work? https://smp.readthedocs.io/en/latest/insights.html#freezing-and-unfreezing-the-encoder
    model = model.to(device)

    transform = sar_transform(resize_size=512,triple=False)

    train_loader, val_loader = get_train_val_dataloaders(
        path_data_dir=TS_PATH,
        batch_size=batch_size, 
        num_workers=0,
        frac_train=0.9,
        transform_train=transform,
        transform_val=transform,
        data_aug=True
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        log_dir=out_log_dir,
        num_epochs=num_epochs,
        unfreeze_backbone=False,
        lr=lr,
        label_smoothing=True,
        weight_decay=1e-4,
        n_tensorboard_plot=8
    )

# tensorboard --logdir=C:\Users\juvad3723\.1\--Projects\Local\seg_test\runs\segformer\512_pruned\vit_0.0001_epochs_160_bs32_WeightedComboLoss --reload_interval 5


