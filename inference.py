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

NUM_EPOCHS = 2
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

test_dataset = Cityscapes(root=root_dir,
                           split='test',
                           mode='fine',
                           target_type='semantic', transform=input_transform,
                           target_transform=target_transform)

test_loader = DataLoader(test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=NUM_WORKERS)


# Initialize U-Net
model = smp.Unet(
    encoder_name="resnet34",        # Choose the encoder. Options include 'resnet34', 'resnet50', etc.
    encoder_weights="imagenet",       # Use pre-trained weights on ImageNet
    in_channels=3,                    # Input channels (3 for RGB images)
    classes=19,                       # Number of output classes for Cityscapes
)

checkpoint_path = "checkpoints/unet_best.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# Helper function: Unnormalize the image
def unnormalize(img_tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1).to(device)
    std = torch.tensor(std).view(-1, 1, 1).to(device)
    img_tensor = img_tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return img_tensor.permute(1, 2, 0).cpu().numpy()

# Obtain one sample from the test loader
# Assume test_loader returns a tuple: (images, masks)
with torch.no_grad():
    images, gt_masks = next(iter(test_loader))
    images = images.to(device)
    gt_masks = gt_masks.to(device)
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

# Select the first sample from the batch
sample_img = images[0]         # shape: [3, H, W]
sample_gt  = gt_masks[0]       # shape: [H, W]
sample_pred = preds[0]         # shape: [H, W]

# Unnormalize the input image for display (assume ImageNet normalization)
unnorm_img = unnormalize(sample_img, mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

# Plot overlay results: one panel for GT overlay and one for Prediction overlay.
plt.figure(figsize=(15, 6))

# Ground Truth Overlay
plt.subplot(1, 2, 1)
plt.imshow(unnorm_img)
plt.imshow(sample_gt.cpu().numpy(), cmap="jet", alpha=0.5)
plt.title("Ground Truth Overlay")
plt.axis("off")

# Prediction Overlay
plt.subplot(1, 2, 2)
plt.imshow(unnorm_img)
plt.imshow(sample_pred.cpu().numpy(), cmap="jet", alpha=0.5)
plt.title("Prediction Overlay")
plt.axis("off")

plt.show()