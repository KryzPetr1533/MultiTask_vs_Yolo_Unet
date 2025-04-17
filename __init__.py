import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import segmentation_models_pytorch as smp
from nuimages.nuimages import NuImages          


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

class NuImagesDataset(Dataset):
    def __init__(self, dataroot, version, transform=None, target_transform=None, sensor_channels=None):
        self.nuim = NuImages(dataroot=dataroot, version=version, lazy=True, verbose=False)
        # Filter for camera keyframes
        self.sd_tokens = [
            sd['token']
            for sd in self.nuim.sample_data
            if sd['is_key_frame'] and
               self.nuim.shortcut('sample_data', 'sensor', sd['token'])['channel'] in (sensor_channels or
                   ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'])
        ]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sd_tokens)

    def __getitem__(self, idx):
        sd_token = self.sd_tokens[idx]
        sample_data = self.nuim.get('sample_data', sd_token)
        img_path = os.path.join(self.nuim.dataroot, sample_data['filename'])
        image = Image.open(img_path).convert('RGB')

        sem_mask, _ = self.nuim.get_segmentation(sd_token) 

        if self.transform:
            image = self.transform(image)
        mask = Image.fromarray(sem_mask.astype(np.uint8))
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


resize_size = (256, 256)
original_size = (1024, 2048)

input_transform = transforms.Compose([
    transforms.Resize(resize_size, interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# Mask transforms (nearest-neighbor resize + to tensor)
target_transform = transforms.Compose([
    transforms.Resize(resize_size, interpolation=Image.NEAREST),
    transforms.PILToTensor(),
    transforms.Lambda(lambda t: t.squeeze(0).long()),
])

# root = os.getenv('NUIMAGES')
root = '/var/tmp/nuImages'
train_version = 'v1.0-mini'
val_version   = 'v1.0-mini'

# Datasets
train_dataset = NuImagesDataset(root, train_version,
                                transform=input_transform,
                                target_transform=target_transform)

val_dataset   = NuImagesDataset(root, val_version,
                                transform=input_transform,
                                target_transform=target_transform)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

val_loader   = DataLoader(val_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=NUM_WORKERS)


for images, masks in train_loader:
    print(images.shape)  # torch.Size([4, 3, H, W])
    print(masks.shape)   # torch.Size([4, H, W])
    print(masks.dtype)   # torch.int64
    break


nuim_train = NuImages(dataroot=root, version=train_version, lazy=True, verbose=False)
num_classes = len(nuim_train.category)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)


from train import train

train(model=model,
      optimizer=optimizer,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=None,       # or define similarly for v1.0-test
      num_epochs=NUM_EPOCHS,
      num_examples=NUM_EXAMPLES,
      scheduler=None,
      freeze_encoder=False)