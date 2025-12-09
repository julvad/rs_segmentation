import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2


def Log_transform(x):
    if isinstance(x, torch.Tensor):
        return torch.log1p(x)
    else:
        return torch.from_numpy(np.log1p(x))

def normalize_tensor(x):
    """Normalize by max value (avoid division by zero)."""
    return x / (x.max() + 1e-6)

def triple_channels(x: torch.Tensor) -> torch.Tensor:
    """Ensure the image has 3 channels (RGB)."""
    return x if x.shape[0] == 3 else x.repeat(3, 1, 1)

def shift_min0(x: torch.Tensor) -> torch.Tensor:
    min_x = x.min()
    return x - min_x if min_x < 0 else x


def basic_transform(resize_size:int = 512, triple:bool = True):
    transforms_list = [
            # transforms.Lambda(shift_min0), # if min<0, shift everything to min==0 ### doesnt work now but OK if ensure positive pixel values
            transforms.Lambda(normalize_tensor), # normalize 0-1
            v2.ToImage(), # convert to 
            v2.Resize((resize_size, resize_size), v2.InterpolationMode.BILINEAR), 
            v2.ToDtype(torch.float32, scale=False)
        ]
    if triple:
        transforms_list.append(transforms.Lambda(triple_channels))

    return transforms.Compose(transforms_list)

def imagenet_transform(resize_size:int = 512):
    """
    Torch transform with ImageNet RGB weights normalization.
    """
    transforms_list = [
            v2.ToImage(), # convert to image
            v2.Resize((resize_size, resize_size), v2.InterpolationMode.BILINEAR), 
            v2.ToDtype(torch.float32, scale=True), 
            transforms.Lambda(triple_channels),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ) # with ImageNet weights 
        ]
    return transforms.Compose(transforms_list)

def sar_transform(resize_size: int = 512, triple:bool=True):
    """
    Torch transform for LINEAR units SAR. with log transform (linear to dB amplitude units)
    """
    transforms_list = [ 
        # transforms.Lambda(shift_min0), # if min<0, shift everything to min==0 ### doesnt work now but OK if ensure positive pixel values
        transforms.Lambda(Log_transform),
        v2.ToImage(), # convert to 
        v2.Resize((resize_size, resize_size), v2.InterpolationMode.BILINEAR), 
        v2.ToDtype(torch.float32, scale=False), 
        transforms.Lambda(normalize_tensor), # normalize 0-1
    ]
    if triple:
        transforms_list.append(transforms.Lambda(triple_channels))

    return transforms.Compose(transforms_list)


### Below: not proof-checked

def oceansar_transform(resize_size: int = 512):
    v2.Compose(
            [
                # v2.Resize((224, 224), v2.InterpolationMode.NEAREST_EXACT), # Original oceansar code
                v2.Resize((resize_size, resize_size), v2.InterpolationMode.BILINEAR), # Bilinear because SAR units? ## Observation: slight improvement
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=False),
                v2.Lambda(
                    lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)
                ),
                v2.Lambda(
                    # lambda x: x / (x.max() # original oceansar code
                    lambda x: x / (x.max() + 1e-6) # missing that small epsilon?
                ),  # Because sometimes the max is o
            ]
        )
    
def dinov3_imagenet_transform(resize_size: int = 512):
    """
    SAR transformations for DINOv3 model. 
    https://github.com/facebookresearch/dinov3?tab=readme-ov-file#image-transforms
    """
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_image = v2.ToImage()
    apply_log = Log_transform()
    to_float = v2.ToDtype(torch.float32, scale=True)
    triple = triple_channels()
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_image, resize, apply_log, to_float, triple, normalize])


def dinov3_sat_transform(resize_size: int = 512):
    """
    SAR transformations for DINOv3 model. 
    https://github.com/facebookresearch/dinov3?tab=readme-ov-file#image-transforms
    """
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_image = v2.ToImage()
    apply_log = Log_transform()
    to_float = v2.ToDtype(torch.float32, scale=True)
    triple = triple_channels()
    normalize = v2.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return v2.Compose([to_image, resize, apply_log, to_float, triple, normalize])

    