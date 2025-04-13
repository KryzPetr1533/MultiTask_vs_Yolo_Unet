import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import segmentation_models_pytorch as smp

from train import train

# import gc

# gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

NUM_EPOCHS = 10
NUM_WORKERS = 2
BATCH_SIZE = 16
NUM_EXAMPLES=1 
FREEZE=False

resize_size = (256, 256)
original_size = (1024, 2048)

input_transform = transforms.Compose([
    transforms.Resize(resize_size, interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def remap_labels(label_tensor):
    # Replace any label that is not in [0, 18] with 255 (the ignore index)
    label_tensor[label_tensor > 18] = 255
    return label_tensor

target_transform = transforms.Compose([
    transforms.Resize(resize_size, interpolation=Image.NEAREST),
    transforms.Lambda(lambda pic: transforms.functional.pil_to_tensor(pic).long().squeeze(0)),
    transforms.Lambda(remap_labels)
])

root_dir = os.getenv('CITYSCAPES_DATASET')

train_dataset = Cityscapes(root=root_dir,
                           split='train',
                           mode='fine',
                           target_type='semantic',
                           transform=input_transform,
                           target_transform=target_transform)

# Create the DataLoader
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

val_dataset = Cityscapes(root=root_dir,
                           split='val',
                           mode='fine',
                           target_type='semantic',
                           transform=input_transform,
                           target_transform=target_transform)

val_loader = DataLoader(val_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

test_dataset = Cityscapes(root=root_dir,
                           split='test',
                           mode='fine',
                           target_type='semantic')

test_loader = DataLoader(test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=NUM_WORKERS)
for images, masks in train_loader:
    print(images.shape)  # torch.Size([4, 3, H, W])
    print(masks.shape)   # torch.Size([4, H, W])
    print(masks.dtype)   # torch.int64
    break


# Initialize U-Net
model = smp.Unet(
    encoder_name="resnet34",        # Choose the encoder. Options include 'resnet34', 'resnet50', etc.
    encoder_weights="imagenet",       # Use pre-trained weights on ImageNet
    in_channels=3,                    # Input channels (3 for RGB images)
    classes=19,                       # Number of output classes for Cityscapes
)

# Optionally move the model to a GPU if available
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model=model, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, num_epochs=NUM_EPOCHS, num_examples=5, scheduler=None, freeze_encoder=False)