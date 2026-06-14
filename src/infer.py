"""
Runs the trained model over a (nested) test dataset directory and writes
predictions.json in the same format as curated_gcp_marks.json.

Run:
    python infer.py --data_root /path/to/test_dataset \
        --weights checkpoints/best_model.pt \
        --out_file predictions.json
"""

import os
import json
import argparse

import cv2
import numpy as np
import torch

from dataset import IDX_TO_SHAPE, IMG_SIZE, get_val_transforms
from model import GCPNet

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_images(root):
    """Recursively find all image files, returning paths relative to root."""
    rel_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(IMAGE_EXTS):
                full = os.path.join(dirpath, fn)
                rel_paths.append(os.path.relpath(full, root))
    return sorted(rel_paths)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GCPNet(backbone=args.backbone)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    transforms = get_val_transforms()
    rel_paths = find_images(args.data_root)
    print(f"Found {len(rel_paths)} test images under {args.data_root}")

    results = {}
    for i, rel_path in enumerate(rel_paths):
        full_path = os.path.join(args.data_root, rel_path)
        img = cv2.imread(full_path)
        if img is None:
            print(f"  WARNING: could not read {full_path}, skipping")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Keypoint placeholder is required by the transform pipeline but
        # unused at inference time.
        transformed = transforms(image=img, keypoints=[(0, 0)])
        img_t = transformed["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_kp, pred_cls = model(img_t)

        kx, ky = pred_kp[0].cpu().numpy()
        cls_idx = int(pred_cls[0].argmax().item())
        shape = IDX_TO_SHAPE[cls_idx]

        # Invert the LongestMaxSize + PadIfNeeded mapping back to original
        # image pixel coordinates.
        scale = IMG_SIZE / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        pad_x = (IMG_SIZE - new_w) / 2.0
        pad_y = (IMG_SIZE - new_h) / 2.0

        orig_x = (kx * IMG_SIZE - pad_x) / scale
        orig_y = (ky * IMG_SIZE - pad_y) / scale

        orig_x = float(np.clip(orig_x, 0, w - 1))
        orig_y = float(np.clip(orig_y, 0, h - 1))

        results[rel_path] = {
            "mark": {"x": orig_x, "y": orig_y},
            "verified_shape": shape,
        }

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(rel_paths)}")

    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote predictions for {len(results)} images to {args.out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Path to test_dataset root")
    p.add_argument("--weights", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--backbone", default="resnet34")
    p.add_argument("--out_file", default="predictions.json")
    args = p.parse_args()
    main(args)
