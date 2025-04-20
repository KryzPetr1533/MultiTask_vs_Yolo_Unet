import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO
from nuimages.nuimages import NuImages
import numpy as np
import os
from collections import defaultdict
from yolo_train import train
import torch.optim as optim
from ultralytics.utils.loss import v8DetectionLoss


class NuImagesYoloDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        version: str,
        classes: list,
        sensor_channels: list = None,
        transform=None,
    ):
        """
        Args:
            dataroot: path to nuImages data
            version: e.g. "v1.0-trainval"
            classes: list of category names to include (maps to class indices 0..N-1)
            sensor_channels: list of camera names to include
            transform: torchvision transforms for the image only
        """
        self.nuim = NuImages(dataroot=dataroot, version=version, lazy=True, verbose=False)
        self.class2idx = {c:i for i,c in enumerate(classes)}
        self.transform = transform

        # 1) collect all keyframe sample_data tokens for chosen cameras
        channels = sensor_channels or ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        self.sd_tokens = [
            sd['token']
            for sd in self.nuim.sample_data
            if sd['is_key_frame'] and
               self.nuim.shortcut('sample_data', 'sensor', sd['token'])['channel'] in channels
        ]

        # 2) build a map from sample_data_token -> list of annotations
        self.annos = defaultdict(list)
        for ann in self.nuim.object_ann:
            sd_tok = ann['sample_data_token']
            if sd_tok in self.sd_tokens:
                # only keep annotations whose category is in our classes list
                cat = self.nuim.get('category', ann['category_token'])['name']
                if cat in self.class2idx:
                    self.annos[sd_tok].append({
                        'bbox': ann['bbox'],        # [x, y, w, h] in pixels
                        'class_id': self.class2idx[cat]
                    })

    def __len__(self):
        return len(self.sd_tokens)

    def __getitem__(self, idx):
        sd_token = self.sd_tokens[idx]
        # --- load image ---
        sd = self.nuim.get('sample_data', sd_token)
        img_path = os.path.join(self.nuim.dataroot, sd['filename'])
        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        # --- load & convert annotations to YOLO format ---
        targets = []
        for ann in self.annos[sd_token]:
            x, y, w, h = ann['bbox']
            # compute centre coords
            x_c = (x + w/2) / W
            y_c = (y + h/2) / H
            w_n = w / W
            h_n = h / H
            targets.append([ann['class_id'], x_c, y_c, w_n, h_n])
        # if no objects, yield an empty tensor of shape (0,5)
        if len(targets):
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = torch.zeros((0,5), dtype=torch.float32)

        # --- apply image transforms (e.g. resize, to_tensor, augment) ---
        if self.transform:
            img = self.transform(img)
        else:
            # convert PIL → Tensor, [0,1]
            img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0

        return img, targets


def yolo_collate_fn(batch):
    """
    Collate for batches of (image, targets), where targets is a variable-length
    Tensor of shape [num_objects,5]. We stack images and keep targets as list.
    """
    imgs = [item[0] for item in batch]
    tgts = [item[1] for item in batch]
    imgs = torch.stack(imgs, dim=0)
    return imgs, tgts

if __name__ == "__main__":
    root, train_ver, val_ver = "/var/tmp/full_nuImages", "v1.0-mini", "v1.0-mini"
    bs, nw = 16, 2

    input_transform = transforms.Compose([
        transforms.Resize((640,640)),
        transforms.ToTensor(),
    ])
    
    classes = [
        "animal",
        "flat.driveable_surface",
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.personal_mobility",
        "human.pedestrian.police_officer",
        "human.pedestrian.stroller",
        "human.pedestrian.wheelchair",
        "movable_object.barrier",
        "movable_object.debris",
        "movable_object.pushable_pullable",
        "movable_object.trafficcone",
        "static_object.bicycle_rack",
        "vehicle.bicycle",
        "vehicle.bus.bendy",
        "vehicle.bus.rigid",
        "vehicle.car",
        "vehicle.construction",
        "vehicle.ego",
        "vehicle.emergency.ambulance",
        "vehicle.emergency.police",
        "vehicle.motorcycle",
        "vehicle.trailer",
        "vehicle.truck"
    ]
    train_ds = NuImagesYoloDataset(root, train_ver, classes=classes, transform=input_transform)
    val_ds   = NuImagesYoloDataset(root, val_ver, classes=classes,  transform=input_transform)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        collate_fn=yolo_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=yolo_collate_fn
    )

    # load YOLOv8 wrapper to get .model with args attached
    yolo = YOLO("yolov8n.pt")
    model = yolo.model  # torch.nn.Module

    # 5) Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 6) Set up optimizer, scheduler, and criterion
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optional: cosine LR that decays over num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    # YOLO loss expects (preds, targets)
    # Ultralytics wrapper provides a built‑in loss inside the model,
    # but if you have a standalone criterion you can pass it here.
    criterion = v8DetectionLoss(model)


    # 7) Train!
    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        # test_loader=test_loader,
        device=device,
        num_epochs=30,
        freeze_backbone=False,
        checkpoint_dir="checkpoints/yolov8_nuimages"
    )

    

    
