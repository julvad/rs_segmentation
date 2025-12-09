import glob
import os
import random
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F

def set_seed(seed: int = 24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class SegmentationDataset(Dataset):
    def __init__(self, list_img_path: list[str], transform: Callable, data_aug:bool=False):
        """
        The data are images stored in int16 and the binary masks stored in int16 with values 0 and 1.

        The data folder should be structured as follows:
            path_folder/
                imgs/
                    1_1.tif
                    1_2.tif
                    ...
                labels/
                    1_1.tif
                    1_2.tif
                    ...

        list_img_path (list[str]): list of paths to the images. We prefer to provide a list of path to the
            images rather than a folder because it is then easier to split the dataset into train and test.
        transform: transform to apply to the images only, not the masks
        """
        self.l_img_path = list_img_path
        self.transform = transform
        self.data_aug = data_aug

    def __len__(self):
        return len(self.l_img_path)

    # def __getitem__(
    #     self, idx: int, plot: bool = False
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     img_path = self.l_img_path[idx]
    #     mask_path = img_path.replace("images", "labels")
    #     if not os.path.exists(mask_path):
    #         raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
    #     img_pil = Image.open(img_path)
    #     with Image.open(mask_path) as img:
    #         mask = np.array(img)


    #     if plot:
    #         plt.figure(figsize=(10, 5))
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(img_pil)
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(mask)
    #         plt.show()

    #     if self.transform:
    #         img = self.transform(img_pil)
    #     mask = torch.from_numpy(mask).long()

    #     return img, mask
    def __getitem__(self, idx: int, plot: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.l_img_path[idx]
        mask_path = img_path.replace("images", "labels")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
        with Image.open(mask_path) as mask_pil:
            mask_tensor = torch.from_numpy(np.array(mask_pil)).long()

        if self.transform:
            with Image.open(img_path) as img_pil:
                img_tensor = self.transform(img_pil)

        # data augmentations (only flips for SAR)
        if self.data_aug:
            if torch.rand(1) < 0.25:
                img_tensor = F.hflip(img_tensor)
                mask_tensor = F.hflip(mask_tensor)
            if torch.rand(1) < 0.25:
                img_tensor = F.vflip(img_tensor)
                mask_tensor = F.vflip(mask_tensor)
        return img_tensor, mask_tensor


def get_train_val_dataloaders(
    path_data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    frac_train: float = 0.9,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    data_aug:bool=False,
    subset: Optional[int] = None,
    return_datasets: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    The structure of {url_data_dir} is the following:
    {url_data_dir}/
        imgs/
            img_0.png
            ...
        masks/
            img_0.png
            ...

    args
        subset (Optional[int]): if specified, only a subset of the images will be used.
    """
    assert os.path.exists(path_data_dir)
    path_folder_imgs = os.path.join(path_data_dir, "images")
    path_folder_masks = os.path.join(path_data_dir, "labels")
    assert os.path.exists(path_folder_imgs)
    assert os.path.exists(path_folder_masks)
    
    set_seed(24)
    l_path_imgs = glob.glob(path_folder_imgs + "/*.tif")
    random.shuffle(l_path_imgs)

    if not subset:
        train_dataset = SegmentationDataset(
            l_path_imgs[: int(len(l_path_imgs) * frac_train)],
            transform=transform_train,
            data_aug=data_aug
        )
        val_dataset = SegmentationDataset(
            l_path_imgs[int(len(l_path_imgs) * frac_train) :],
            transform=transform_val,
            data_aug=data_aug
        )
    else:
        min_subset = min(subset, len(l_path_imgs))
        train_dataset = SegmentationDataset(
            l_path_imgs[: int(min_subset * frac_train)],
            transform=transform_train,
            data_aug=data_aug
        )
        val_dataset = SegmentationDataset(
            l_path_imgs[int(min_subset * frac_train) :],
            transform=transform_val,
            data_aug=data_aug
        )

    # Very standard dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    if not return_datasets:
        return train_loader, val_loader
    else:
        return train_dataset, val_dataset
