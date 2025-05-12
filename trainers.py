import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from nuimages.nuimages import NuImages
from tqdm import tqdm

def process_sample(job, nuim_dataroot, output_label_dir, output_image_dir, cat_token_to_id):
    """
    job is a tuple:
      (token, rel_path, width, height, anns_for_token)
    """
    token, rel_path, w, h, anns = job

    # Prepare filesystem paths
    base = pathlib.Path(rel_path).stem
    label_path = pathlib.Path(output_label_dir) / f"{base}.txt"
    img_link   = pathlib.Path(output_image_dir) / pathlib.Path(rel_path).name
    abs_src    = pathlib.Path(nuim_dataroot) / rel_path

    # Create a symlink if it doesn't already exist
    try:
        img_link.symlink_to(abs_src.resolve())
    except FileExistsError:
        pass

    # Build YOLO lines
    lines = []
    for ann in anns:
        xmin, ymin, xmax, ymax = ann['bbox']
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        bw = (xmax - xmin)
        bh = (ymax - ymin)
        cid = cat_token_to_id[ann['category_token']]
        lines.append(f"{cid} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")

    # Write label file in one go
    label_path.write_text("\n".join(lines))

def main(version='v1.0-mini', split='train', num_workers=None):
    nuim = NuImages(
        dataroot='/var/tmp/full_nuImages',
        version=version,
        verbose=True
    )
    root = pathlib.Path(nuim.dataroot) / version
    label_dir = root / 'labels' / split
    image_dir = root / 'images' / split
    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pre-group all annotations by sample_data_token
    anns_by_token = {}
    for ann in nuim.object_ann:
        anns_by_token.setdefault(ann['sample_data_token'], []).append(ann)

    # 2) Pre-map category_token → class_id
    cat_token_to_id = {
        cat['token']: idx
        for idx, cat in enumerate(nuim.category)
    }

    # 3) Build a lightweight job list: no NuImages calls inside workers
    jobs = []
    for sample in nuim.sample:
        token = sample['key_camera_token']
        sd = nuim.get('sample_data', token)
        jobs.append((
            token,
            sd['filename'],
            sd['width'],
            sd['height'],
            anns_by_token.get(token, [])
        ))

    # 4) Parallelize over CPU cores
    num_workers = num_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [
            exe.submit(
                process_sample,
                job,
                nuim.dataroot,
                str(label_dir),
                str(image_dir),
                cat_token_to_id
            )
            for job in jobs
        ]
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc=f"Writing {split}"):
            pass

if __name__ == '__main__':
    # Write train labels
    main(version='v1.0-train', split='train')
    # Write val labels
    main(version='v1.0-val',   split='val')
