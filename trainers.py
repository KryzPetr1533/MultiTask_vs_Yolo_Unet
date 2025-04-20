from nuimages.nuimages import NuImages
import os
import cv2
from tqdm import tqdm

def main(version='v1.0-mini', split='train'):
    # Initialize nuImages
    nuim = NuImages(
        dataroot='/var/tmp/full_nuImages',
        version=version,
        verbose=True
    )

    # Prepare output dirs
    output_label_dir = f'/var/tmp/full_nuImages/{version}/labels/{split}/'
    output_image_dir = f'/var/tmp/full_nuImages/{version}/images/{split}/'
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    # Category → ID map
    category_to_id = {
        cat['name']: idx
        for idx, cat in enumerate(nuim.category)
    }

    # Iterate through all samples
    for sample in tqdm(nuim.sample, desc=f'Writing {split} labels'):
        key_camera_token = sample['key_camera_token']
        sample_data = nuim.get('sample_data', key_camera_token)
        image_rel_path = sample_data['filename']
        image_path = os.path.join(nuim.dataroot, image_rel_path)

        # Skip missing / unreadable images
        if not os.path.exists(image_path):
            continue
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w, _ = img.shape

        # Prepare label file path
        base = os.path.splitext(os.path.basename(image_rel_path))[0]
        label_filepath = os.path.join(output_label_dir, base + '.txt')

        # Symlink image into YOLO images folder
        link_img_path = os.path.join(output_image_dir, os.path.basename(image_rel_path))
        if not os.path.exists(link_img_path):
            os.symlink(os.path.abspath(image_path), link_img_path)

        # Collect all object annotations for this sample
        anns = [
            ann for ann in nuim.object_ann
            if ann['sample_data_token'] == key_camera_token
        ]

        # Write YOLO-format labels
        with open(label_filepath, 'w') as f:
            for ann in anns:
                xmin, ymin, xmax, ymax = ann['bbox']

                # Convert [xmin, ymin, xmax, ymax] → [cx, cy, w, h]
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0
                bw = xmax - xmin
                bh = ymax - ymin

                # Normalize
                cx_norm = cx / w
                cy_norm = cy / h
                bw_norm = bw / w
                bh_norm = bh / h

                # Lookup class ID
                category = nuim.get('category', ann['category_token'])['name']
                class_id = category_to_id[category]

                f.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} "
                        f"{bw_norm:.6f} {bh_norm:.6f}\n")

if __name__ == '__main__':
    version = 'v1.0-mini'
    split = 'val'

    # Write train labels
    main(version=version, split='train')
    # Write val labels
    main(version=version, split=split)
