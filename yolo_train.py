import os
import sys
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Any, Dict
from torch import nn
from torch.utils.data import DataLoader, Dataset
from IPython.display import clear_output
from tqdm.notebook import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import gc

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})

def plot_losses(train_losses, val_losses):
    """
    Plot training vs. validation loss curves.
    
    :param train_losses: List[float], loss at each epoch on training set
    :param val_losses:   List[float], loss at each epoch on validation set
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses,   marker='o', label='Train Loss')
    plt.plot(epochs, val_losses,     marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def training_epoch(
    det_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    freeze_backbone: bool = False,
    desc: str = "Training"
) -> float:
    det_model.train()
    criterion = det_model.init_criterion()  # v8DetectionLoss(self)
    running_loss = 0.0

    # optional backbone freeze
    if freeze_backbone:
        for _, p in det_model.backbone.named_parameters():
            p.requires_grad = False
    else:
        for p in det_model.parameters():
            p.requires_grad = True

    for images, targets in tqdm(dataloader, desc=desc):
        images = images.to(device)

        # 1) Gather all labels into one [N,6] tensor
        labels = []
        for batch_i, t in enumerate(targets):
            if t.numel():
                idx_col = torch.full((t.size(0),1), batch_i, device=device)
                labels.append(torch.cat([idx_col, t.to(device)], dim=1))
        labels = torch.cat(labels, 0) if labels else torch.zeros((0,6), device=device)

        # 2) Split into the dict that v8DetectionLoss wants
        batch_dict = {
            "batch_idx": labels[:, 0].long(),
            "cls":       labels[:, 1].long(),
            "bboxes":    labels[:, 2:].float(),
        }

        optimizer.zero_grad()
        preds = det_model(images)               # forward pass → list of pred tensors
        loss = criterion(preds, batch_dict)     # scalar
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)



@torch.no_grad()
def validation_epoch(
    det_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "Validation"
) -> float:
    det_model.eval()
    criterion = det_model.init_criterion()
    running_loss = 0.0

    for images, targets in tqdm(dataloader, desc=desc):
        images = images.to(device)

        labels = []
        for batch_i, t in enumerate(targets):
            if t.numel():
                idx_col = torch.full((t.size(0),1), batch_i, device=device)
                labels.append(torch.cat([idx_col, t.to(device)], dim=1))
        labels = torch.cat(labels, 0) if labels else torch.zeros((0,6), device=device)

        batch_dict = {
            "batch_idx": labels[:, 0].long(),
            "cls":       labels[:, 1].long(),
            "bboxes":    labels[:, 2:].float(),
        }

        preds = det_model(images)
        loss = criterion(preds, batch_dict)
        running_loss += loss.item()

    return running_loss / len(dataloader)


def train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_epochs: int,
        freeze_backbone: bool = False,
        checkpoint_dir: str = "checkpoints",
        test_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
    """
    Train a YOLO‐style detector for several epochs, with validation and final test‐loss.
    
    :param model:           The YOLO detection model.
    :param optimizer:       Optimizer (e.g., Adam or SGD).
    :param scheduler:       Optional LR scheduler (should call .step()).
    :param criterion:       YOLO loss module (can return scalar or dict of losses).
    :param train_loader:    DataLoader for training (yields images, targets).
    :param val_loader:      DataLoader for validation.
    :param test_loader:     DataLoader for test (only used for final evaluation).
    :param device:          torch.device ("cuda" or "cpu").
    :param num_epochs:      Total epochs to train.
    :param freeze_backbone: If True, freezes model.backbone parameters during training.
    :param checkpoint_dir:  Directory to save best/last checkpoints.
    :param disable_tqdm:    If True, disables tqdm bars.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        # clear GPU memory if available
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        # Training step
        train_loss = training_epoch(
            model,
            optimizer,
            train_loader,
            device=device,
            freeze_backbone=freeze_backbone,
            desc=f"Epoch {epoch}/{num_epochs} [Train]",
        )

        # Validation step
        val_loss = validation_epoch(
            model,
            val_loader,
            device=device,
            desc=f"Epoch {epoch}/{num_epochs} [Val]",
            )

        # Step LR scheduler if provided
        if scheduler is not None:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:2d} ▶ "
            f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(checkpoint_dir, "yolo_best.pt")
            torch.save(model.state_dict(), path)

    # Save final model
    last_path = os.path.join(checkpoint_dir, "yolo_last.pt")
    torch.save(model.state_dict(), last_path)

    # Plot loss curves
    plot_losses(train_losses, val_losses)

    # Final test‐set loss
    if test_loader is not None:
        test_loss = validation_epoch(
            model,
            test_loader,
            device=device,
            desc="Test",
        )
        print(f"Test_loss: {test_loss:.4f}")