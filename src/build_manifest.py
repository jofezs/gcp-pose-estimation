"""
Builds train/val manifests from gcp_marks.json.

Split strategy: by GCP folder (project/survey/gcp_id), NOT by individual
image and NOT purely by project. Images within the same GCP folder are
highly correlated (same physical marker), so we keep each GCP folder
entirely in train or entirely in val. This prevents leakage while still
allowing all 3 shape classes to appear in both splits.
"""

import os
import json
import random
import argparse
from collections import defaultdict

SHAPE_NORMALIZE = {
    "Cross": "Cross",
    "Square": "Square",
    "L-Shape": "L-Shape",
    "L-Shaped": "L-Shape",
}


def main(args):
    with open(args.labels_json) as f:
        labels = json.load(f)

    samples = []
    skipped = []

    for rel_path, label in labels.items():
        full_path = os.path.join(args.data_root, rel_path)
        if not os.path.exists(full_path):
            continue
        raw_shape = label.get("verified_shape")
        if raw_shape not in SHAPE_NORMALIZE:
            skipped.append((rel_path, raw_shape))
            continue
        label = dict(label)
        label["verified_shape"] = SHAPE_NORMALIZE[raw_shape]
        samples.append((rel_path, label))

    print(f"Usable samples: {len(samples)}  |  Skipped (bad label): {len(skipped)}")

    # Group by GCP folder: parts[0]/parts[1]/parts[2]
    gcp_groups = defaultdict(list)
    gcp_shape = {}  # dominant shape of each GCP folder
    for rel_path, label in samples:
        parts = rel_path.replace("\\", "/").split("/")
        gcp_key = "/".join(parts[:3])
        gcp_groups[gcp_key].append((rel_path, label))
        gcp_shape[gcp_key] = label["verified_shape"]

    # Separate GCP keys by shape class
    by_class = defaultdict(list)
    for gcp_key, shape in gcp_shape.items():
        by_class[shape].append(gcp_key)

    print(f"GCP folders per class: { {k: len(v) for k, v in by_class.items()} }")

    random.seed(args.seed)
    val_gcps = set()

    # Sample ~val_fraction of GCP folders from EACH class independently
    # so all 3 classes are guaranteed in val.
    for shape, keys in by_class.items():
        random.shuffle(keys)
        n_val = max(1, round(len(keys) * args.val_fraction))
        val_gcps.update(keys[:n_val])

    train_samples, val_samples = [], []
    for gcp_key, items in gcp_groups.items():
        if gcp_key in val_gcps:
            val_samples.extend(items)
        else:
            train_samples.extend(items)

    print(f"Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    for name, split in [("train", train_samples), ("val", val_samples)]:
        counts = defaultdict(int)
        for _, label in split:
            counts[label["verified_shape"]] += 1
        print(f"  {name} class distribution: {dict(counts)}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train_manifest.json"), "w") as f:
        json.dump(train_samples, f)
    with open(os.path.join(args.out_dir, "val_manifest.json"), "w") as f:
        json.dump(val_samples, f)

    print(f"\nWrote manifests to {args.out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--labels_json", required=True)
    p.add_argument("--out_dir", default="manifests")
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)