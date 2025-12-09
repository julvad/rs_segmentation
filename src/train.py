import json
import os

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.utils.data as DataLoader
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from utils.utils_plot import plot_tensorboard_batch_images


def save_and_manage_checkpoints(
    val_loss, iou_val, dice_val, epoch, model, optimizer, log_dir, best_models, max_models=3
):
    model_path = os.path.join(
        log_dir, f"best_model_epoch_{epoch + 1}_{val_loss:.4f}"
    )
    if 'dinov3' in log_dir:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            model_path,
        )
    else:
        model.save_pretrained(
            model_path, 
            metrics={'epoch':epoch+1, 'iou': iou_val, 'dice': dice_val}, 
            dataset='512_20_pruned3570'
            )
    best_models.append((val_loss, epoch, model_path))
    # Sort and keep only the best max_models
    best_models.sort(key=lambda x: x[0])
    while len(best_models) > max_models:
        worst = best_models.pop()  # Remove the worst (highest val_loss)
        try:
            if os.path.isdir(worst[2]):
                os.rmdir(worst[2])
            else:
                os.remove(worst[2])
            print(f"Removed old checkpoint: {worst[2]}")
        except Exception as e:
            print(f"Could not remove file: {worst[2]} ({e})")
    return best_models


# Dice Loss for binary segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] (raw model outputs, C=2 for binary segmentation)
        targets: [B, H, W] (class indices, 0=background, 1=foreground)
        """
        # Use foreground channel only
        probs = torch.softmax(logits, dim=1)[:, 1, ...]  # [B,H,W]
        targets = targets.float()

        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss.mean()


# Combo Loss = α * CE + (1 - α) * Dice
class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0):
        """
        alpha: weight for CE; (1 - alpha) is weight for Dice
        """
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)  # targets: [B,H,W] with class indices
        dice_loss = self.dice(logits, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss
    
def compute_class_weights(dataloader, num_classes):
    print(f'Computing class weights... for {num_classes} classes')
    total_pixels = torch.zeros(num_classes)
    for _, targets in dataloader:
        # targets: [B, H, W]
        for c in range(num_classes):
            total_pixels[c] += (targets == c).sum()
    weights = 1.0 / (total_pixels + 1e-6)
    weights = weights / weights.sum() * num_classes  # normalize to keep scale consistent
    return weights


class WeightedComboLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 25,
    lr: float = 1e-2,
    num_classes: int = 2,
    weight_decay: float = 1e-4,
    log_dir: str = "runs/segmentation_experiment",
    unfreeze_backbone: bool = False,
    type_scheduler: str = "reduce_lr",
    label_smoothing: bool = True,
    device: str = "cuda:0",
    n_tensorboard_plot:int=16
) -> None:
    """
    Here,

    args:
    - data_mode (str) = 'pancro_duplication',
    - unfreeze_backbone: bool = False,
    num_classes (int): 4 or 6.
    - type_similarity (str): the type of similarity matrix we want to use. If 'transition', it means
        that it encodes the similarity in terms of how hard it is to transition from one class to the other
        and 'impossibility' encodes how two blobs can be different from one another ?
    """

    assert num_classes == 2
    assert type_scheduler in ["reduce_lr", "cosine_annealing"]

    os.makedirs(log_dir, exist_ok=True)

## LOSS functions
#______________________________________Weighted combo loss
    # class_weights = compute_class_weights(train_loader, num_classes=2)
    # print("Class weights:", class_weights)
    # criterion = WeightedComboLoss(alpha=0.5, smooth=1.0,class_weights=class_weights.to(device))
#_______________________________________ Weighted CE
    # class_weights = compute_class_weights(train_loader, num_classes=2)
    # print("Class weights:", class_weights)
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion = nn.CrossEntropyLoss()
    if label_smoothing:
        criterion= nn.CrossEntropyLoss(label_smoothing=0.1)
#_________________________________________________________ ComboLoss (not weighted)
    # criterion = ComboLoss(alpha=0.5, smooth=1.0)
#__________________________________________________________



    writer = SummaryWriter(log_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if type_scheduler == "reduce_lr":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.99
        )
    elif type_scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    else:
        raise ValueError(f"Scheduler type {type_scheduler} not yet supported")

    if unfreeze_backbone:
        for param in model.parameters():
            param.requires_grad = True

    # Print the number of trainable parameters:
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # On initialise la meilleure validation loss
    best_val_loss = float("inf")
    best_models = []


    iou_train = MulticlassJaccardIndex(num_classes=2, average="macro").to(device)
    dice_train = MulticlassF1Score(num_classes=2, average="macro").to(device)
    iou_val = MulticlassJaccardIndex(num_classes=2, average="macro").to(device)
    dice_val = MulticlassF1Score(num_classes=2, average="macro").to(device)

    global_step = 0
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        iou_train.reset()
        dice_train.reset()
        train_loss = 0

        for images, masks in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            iou_train.update(outputs, masks)
            dice_train.update(outputs, masks)

            # Log training loss every 25 batches (or 200)
            if global_step % 200 == 0:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)
                writer.add_scalar(
                    "IOU/train_step", iou_train.compute().item(), global_step
                )
                writer.add_scalar(
                    "Dice/train_step", dice_train.compute().item(), global_step
                )

            global_step += 1

        avg_train_loss = train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        l_predicted_masks = []
        l_masks = []
        l_images = []
        iou_val.reset()
        dice_val.reset()
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)

                loss = criterion(outputs, masks)

                val_loss += loss.item()

                predicted_mask = outputs.argmax(dim=1)
                            
                l_predicted_masks.append(predicted_mask.cpu())
                l_masks.append(masks.cpu())
                l_images.append(images.cpu())

                iou_val.update(outputs, masks)
                dice_val.update(outputs, masks)

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        all_images = torch.cat(l_images, dim=0)
        all_masks = torch.cat(l_masks, dim=0)
        all_predicted_masks = torch.cat(l_predicted_masks, dim=0)
        plot_tensorboard_batch_images(
            writer,
            all_images,
            all_masks,
            all_predicted_masks,
            epoch,
            name_tensorboard="val",
            n_images=n_tensorboard_plot
        )

        iou_val_epoch = iou_val.compute().item()
        dice_val_epoch = dice_val.compute().item()

        # Log epoch metrics
        writer.add_scalar("IOU/val_epoch", iou_val_epoch, epoch)
        writer.add_scalar("Dice/val_epoch", dice_val_epoch, epoch)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print('iou_val:', iou_val_epoch)
        print('dice_val:', dice_val_epoch)

        if avg_val_loss < best_val_loss or len(best_models) < 4:
            best_models = save_and_manage_checkpoints(
                avg_val_loss,
                iou_val_epoch,
                dice_val_epoch,
                epoch,
                model,
                optimizer,
                log_dir,
                best_models,
                max_models=4,
            )
            best_val_loss = min([x[0] for x in best_models])
        del l_images, l_masks, l_predicted_masks, all_images, all_masks, all_predicted_masks
        torch.cuda.empty_cache()

    writer.close()

    # Save a simple json with the val loss
    url_json_file = os.path.join(log_dir, "model_metadata.json")
    with open(url_json_file, "w") as f:
        json.dump(
            {
                "val_loss": best_val_loss,
                "data_mode": "colorized_images",
                "unfreeze_backbone": unfreeze_backbone,
                "num_epochs": num_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "type_scheduler": type_scheduler,
            },
            f,
        )
    
    
    del outputs, predicted_mask, images, masks
    torch.cuda.empty_cache()

