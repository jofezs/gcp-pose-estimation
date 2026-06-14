"""
Exploratory Data Analysis for the GCP dataset.

Run:
    python eda.py --data_root /path/to/train_dataset --labels_json /path/to/curated_gcp_marks.json

This script is intentionally a plain script (not a notebook) so it can be
run headlessly against the real dataset; convert to a notebook if you want
inline plots.
"""

import os
import json
import argparse
from collections import Counter, defaultdict

import cv2


def main(args):
    with open(args.labels_json) as f:
        labels = json.load(f)

    print(f"Total annotated entries: {len(labels)}")

    raw_shape_names = Counter()
    img_sizes = Counter()
    aspect_ratios = Counter()
    coord_outside = 0
    missing = 0
    dup_paths = Counter()

    for rel_path, label in labels.items():
        dup_paths[rel_path] += 1
        raw_shape_names[label.get("verified_shape")] += 1

        full_path = os.path.join(args.data_root, rel_path)
        if not os.path.exists(full_path):
            missing += 1
            continue

        img = cv2.imread(full_path)
        if img is None:
            missing += 1
            continue

        h, w = img.shape[:2]
        img_sizes[(w, h)] += 1
        aspect_ratios[round(w / h, 3)] += 1

        x, y = label["mark"]["x"], label["mark"]["y"]
        if not (0 <= x <= w and 0 <= y <= h):
            coord_outside += 1

    print("\n--- Shape label distribution (raw values as found in JSON) ---")
    for k, v in raw_shape_names.items():
        print(f"  {k!r}: {v}")

    print("\n--- Image resolutions found ---")
    for size, count in img_sizes.most_common():
        print(f"  {size[0]}x{size[1]}: {count}")

    print("\n--- Aspect ratios found ---")
    for ar, count in aspect_ratios.most_common():
        print(f"  {ar}: {count}")

    print(f"\nMissing / unreadable images: {missing}")
    print(f"Annotations with (x, y) outside image bounds: {coord_outside}")

    duplicates = {k: v for k, v in dup_paths.items() if v > 1}
    print(f"\nDuplicate JSON keys (should be 0, JSON keys are unique by definition): {len(duplicates)}")

    # Project-level breakdown — useful for designing a leakage-free split.
    projects = defaultdict(int)
    surveys = defaultdict(set)
    for rel_path in labels:
        parts = rel_path.split("/")
        project = parts[0]
        survey = "/".join(parts[:2])
        projects[project] += 1
        surveys[project].add(survey)

    print(f"\nDistinct top-level projects: {len(projects)}")
    for proj, count in sorted(projects.items(), key=lambda x: -x[1]):
        print(f"  {proj}: {count} images across {len(surveys[proj])} survey folder(s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Path to train_dataset root")
    p.add_argument("--labels_json", required=True, help="Path to curated_gcp_marks.json")
    args = p.parse_args()
    main(args)
