import sys
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
# import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import gc

gc.collect()
torch.cuda.empty_cache()

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})

def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """ 
    # clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    # Calculate train and validation perplexities given lists of losses.
    # For language models, perplexity is usually defined as the exponentiation
    # of the cross-entropy loss.
    train_perplexities = [np.exp(loss) for loss in train_losses]
    val_perplexities = [np.exp(loss) for loss in val_losses]

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()



def training_epoch(model, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                loader: DataLoader, tqdm_desc: str, freeze_encoder: bool):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0
    model.train()

    for param in model.parameters():
        param.requires_grad = True

    if freeze_encoder:
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad  = False

    disable_tqdm = not sys.stdout.isatty()

    for indices, (images, masks) in enumerate(tqdm(loader, desc=tqdm_desc, disable=disable_tqdm)):
        """
        Process one training step: calculate loss,
        call backward and make one optimizer step.
        Accumulate sum of losses for different batches in train_loss
        """
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model, criterion: nn.Module,
                    loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    disable_tqdm = not sys.stdout.isatty()
    
    model.eval()
    for indices, (images, masks) in enumerate(tqdm(loader, desc=tqdm_desc, disable=disable_tqdm)):
        """
        Process one validation step: calculate loss.
        Accumulate sum of losses for different batches in val_loss
        """

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        val_loss += loss.item()


    val_loss /= len(loader.dataset)
    return val_loss



def train(model, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
        train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, num_epochs: int, num_examples=5, freeze_encoder=False):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    device = next(model.parameters()).device

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}', freeze_encoder=freeze_encoder
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
    plot_losses(train_losses, val_losses)

 
    # model.eval()
    # with torch.no_grad():
    #     for image, mask in test_loader:  # Assuming test_loader returns PIL Images or similar
    #         # If test_loader returns a tuple (image, …) then extract the image appropriately.
    #         # For example: image, _ = image_tuple
    #         inference_transform = transforms.Compose([
    #             transforms.Resize((256, 256), interpolation=Image.BILINEAR),  # if needed
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                                 std=[0.229, 0.224, 0.225])
    #         ])
    #         # Apply minimal preprocessing
    #         img_tensor = inference_transform(image).unsqueeze(0).to(device)
            
    #         # Forward pass
    #         output = model(img_tensor)
    #         pred_mask = torch.argmax(output, dim=1).squeeze(0)  # shape: (H, W)
            
    #         # Optionally, you can resize the prediction back to the original size if needed
    #         # For full-resolution display, use F.interpolate with nearest mode:
    #         orig_size = image.size[::-1]  # PIL image size returns (width, height)
    #         pred_mask_resized = torch.nn.functional.interpolate(
    #               pred_mask.unsqueeze(0).unsqueeze(0).float(), size=orig_size, mode='nearest'
    #         ).squeeze()
            
    #         fig, axs = plt.subplots(1, 2, figsize=(15, 10))
            
    #         # Ground Truth Overlay
    #         axs[0].imshow(image)
    #         axs[0].imshow(mask, cmap='jet', alpha=0.5)
    #         axs[0].set_title('Ground Truth Overlay')
    #         axs[0].axis('off')
            
    #         # Prediction Overlay
    #         axs[1].imshow(image)
    #         axs[1].imshow(pred_mask_resized.cpu().numpy(), cmap='jet', alpha=0.5)
    #         axs[1].set_title('Prediction Overlay')
    #         axs[1].axis('off')
            
    #         plt.show()
