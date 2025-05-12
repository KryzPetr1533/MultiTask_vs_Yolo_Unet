import os
import sys
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Any, Dict
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from IPython.display import clear_output
import torchvision.transforms as transforms
from PIL import Image

# Label used to ignore invalid classes in loss
IGNORE_INDEX = 255

def compute_segmentation_metrics(outputs: torch.Tensor,
                                 masks: torch.Tensor,
                                 num_classes: int) -> Dict[str, float]:
    preds = torch.argmax(outputs, dim=1)
    # Exclude ignored pixels from metric computation
    valid_mask = masks != IGNORE_INDEX
    running_iou, running_dice = 0.0, 0.0
    for cls in range(num_classes):
        pred_cls = (preds == cls) & valid_mask
        mask_cls = (masks == cls)
        inter = (pred_cls & mask_cls).sum().item()
        union = (pred_cls | mask_cls).sum().item()
        iou = 1.0 if union == 0 else inter / union
        dice = (2 * inter) / (pred_cls.sum().item() + mask_cls.sum().item() + 1e-6)
        running_iou += iou
        running_dice += dice
    mean_iou = running_iou / num_classes
    mean_dice = running_dice / num_classes
    correct = ((preds == masks) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    pixel_acc = correct / total if total > 0 else 0.0
    return {"pixel_acc": pixel_acc, "mean_iou": mean_iou, "mean_dice": mean_dice}


def plot_losses(train_losses: List[float], val_losses: List[float]):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='train_loss')
    ax.plot(epochs, val_losses, label='val_loss')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CrossEntropy Loss')
    ax.legend()
    plt.show()


def plot_metrics(metrics_history: Dict[str, Dict[str, List[float]]]):
    sns.set_style('whitegrid')
    metrics = ['pixel_acc', 'mean_iou', 'mean_dice']
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    epochs = range(1, len(metrics_history['train']['pixel_acc']) + 1)
    for idx, m in enumerate(metrics):
        ax = axes[idx]
        ax.plot(epochs, metrics_history['train'][m], label='train')
        ax.plot(epochs, metrics_history['val'][m], label='val')
        ax.set_title(m.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.legend()
    plt.show()


def training_epoch(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   loader: DataLoader,
                   num_classes: int,
                   tqdm_desc: str,
                   freeze_encoder: bool) -> (float, Dict[str, float]):
    device = next(model.parameters()).device
    model.train()
    # Unfreeze all parameters, optionally freeze encoder
    for p in model.parameters(): p.requires_grad = True
    if freeze_encoder:
        for name, p in model.named_parameters():
            if 'encoder' in name:
                p.requires_grad = False

    running_loss = 0.0
    running_metrics = {k: 0.0 for k in ['pixel_acc', 'mean_iou', 'mean_dice']}
    total_batches = 0
    disable_tqdm = not sys.stdout.isatty()
    for images, masks in tqdm(loader, desc=tqdm_desc, disable=disable_tqdm):
        images, masks = images.to(device), masks.to(device)
        # Remap any invalid mask labels to IGNORE_INDEX
        masks = masks.clone()
        masks[(masks < 0) | (masks >= num_classes)] = IGNORE_INDEX

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_metrics = compute_segmentation_metrics(outputs, masks, num_classes)
        for k in running_metrics:
            running_metrics[k] += batch_metrics[k]
        total_batches += 1

    avg_loss = running_loss / len(loader.dataset)
    for k in running_metrics:
        running_metrics[k] /= total_batches
    return avg_loss, running_metrics


@torch.no_grad()
def validation_epoch(model: nn.Module,
                     criterion: nn.Module,
                     loader: DataLoader,
                     num_classes: int,
                     tqdm_desc: str) -> (float, Dict[str, float]):
    device = next(model.parameters()).device
    model.eval()
    running_loss = 0.0
    running_metrics = {k: 0.0 for k in ['pixel_acc', 'mean_iou', 'mean_dice']}
    total_batches = 0
    disable_tqdm = not sys.stdout.isatty()
    for images, masks in tqdm(loader, desc=tqdm_desc, disable=disable_tqdm):
        images, masks = images.to(device), masks.to(device)
        # Remap invalid labels
        masks = masks.clone()
        masks[(masks < 0) | (masks >= num_classes)] = IGNORE_INDEX

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item()
        batch_metrics = compute_segmentation_metrics(outputs, masks, num_classes)
        for k in running_metrics:
            running_metrics[k] += batch_metrics[k]
        total_batches += 1

    avg_loss = running_loss / len(loader.dataset)
    for k in running_metrics:
        running_metrics[k] /= total_batches
    return avg_loss, running_metrics


def train(model: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: Optional[Any],
          train_loader: DataLoader,
          val_loader: DataLoader,
          test_loader: DataLoader,
          num_epochs: int,
          num_classes: int,
          checkpoint_dir: str = './checkpoints',
          freeze_encoder: bool = False,
          plot_every: int = 1):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_miou = -float('inf')

    train_losses, val_losses = [], []
    metrics_history = {
        'train': {k: [] for k in ['pixel_acc', 'mean_iou', 'mean_dice']},
        'val':   {k: [] for k in ['pixel_acc', 'mean_iou', 'mean_dice']}
    }
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    for epoch in tqdm(range(1, num_epochs + 1), desc='Epochs'):
        tl, tm = training_epoch(
            model, optimizer, criterion, train_loader,
            num_classes, f'Train {epoch}/{num_epochs}', freeze_encoder
        )
        vl, vm = validation_epoch(
            model, criterion, val_loader,
            num_classes, f'Val {epoch}/{num_epochs}'
        )
        if scheduler:
            try:
                scheduler.step(vm['mean_iou'])
            except Exception:
                scheduler.step()

        train_losses.append(tl)
        val_losses.append(vl)
        for k in metrics_history['train']:
            metrics_history['train'][k].append(tm[k])
            metrics_history['val'][k].append(vm[k])

        print(f"Epoch {epoch}: train_loss={tl:.4f}, val_loss={vl:.4f}, "
              f"train_mIoU={tm['mean_iou']:.4f}, val_mIoU={vm['mean_iou']:.4f}")

        last_path = os.path.join(checkpoint_dir, 'u-net-last.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }, last_path)
        if vm['mean_iou'] > best_miou:
            best_miou = vm['mean_iou']
            best_path = os.path.join(checkpoint_dir, 'u-net-best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_miou': best_miou
            }, best_path)
            print(f"Best saved to {best_path} (mIoU={best_miou:.4f})")

        if epoch % plot_every == 0 or epoch == num_epochs:
            clear_output()
            plot_losses(train_losses, val_losses)
            plot_metrics(metrics_history)

    return model
