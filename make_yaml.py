#!/usr/bin/env python3
import os
from pathlib import Path
import yaml
from nuimages.nuimages import NuImages

def make_nuimages_yaml(root, out_yaml="nuimages.yaml"):
    # 1. Gather train & val image paths from separate versions
    def get_paths(version):
        nuim = NuImages(dataroot=root, version=version, lazy=True, verbose=False)
        return [
            os.path.join(root, sd['filename'])
            for sd in nuim.sample_data
            if sd['is_key_frame']
            and nuim.shortcut('sample_data','sensor',sd['token'])['channel'].startswith("CAM_")
        ]

    train_imgs = get_paths("v1.0-train")
    val_imgs   = get_paths("v1.0-val")

    # 2. Write lists to .txt
    os.makedirs("splits", exist_ok=True)
    Path("splits/train.txt").write_text("\n".join(train_imgs))
    Path("splits/val.txt").write_text("\n".join(val_imgs))

    # 3. Build YAML config
    nuim_train = NuImages(dataroot=root, version="v1.0-train", lazy=True, verbose=False)
    names = [c['name'] for c in nuim_train.category]
    cfg = {
        'path': root,
        'train': 'splits/train.txt',  # list-of-files mode :contentReference[oaicite:6]{index=6}
        'val':   'splits/val.txt',
        'nc':    len(names),
        'names': names
    }

    # 4. Dump to nuimages.yaml
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"→ {out_yaml} with {len(train_imgs)} train / {len(val_imgs)} val images")

if __name__ == "__main__":
    make_nuimages_yaml("/var/tmp/full_nuImages")
