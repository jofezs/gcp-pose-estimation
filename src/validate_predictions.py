"""
Validates predictions.json format and sanity-checks the outputs.

Run:
    python validate_predictions.py --predictions predictions.json \
        --test_root /path/to/test_dataset
"""

import os
import json
import argparse
from collections import Counter

VALID_SHAPES = {"Cross", "Square", "L-Shape"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_images(root):
    rel_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(IMAGE_EXTS):
                full = os.path.join(dirpath, fn)
                rel_paths.append(os.path.relpath(full, root).replace("\\", "/"))
    return set(rel_paths)


def main(args):
    with open(args.predictions) as f:
        preds = json.load(f)

    print(f"Total predictions: {len(preds)}")

    # Find actual test images
    test_images = find_images(args.test_root)
    print(f"Total test images on disk: {len(test_images)}")

    errors = []
    shape_counts = Counter()
    missing_from_preds = []
    extra_in_preds = []

    # Normalize path separators
    pred_keys = {k.replace("\\", "/"): v for k, v in preds.items()}

    for img_path in test_images:
        if img_path not in pred_keys:
            missing_from_preds.append(img_path)

    for pred_path, pred in pred_keys.items():
        if pred_path not in test_images:
            extra_in_preds.append(pred_path)
            continue

        # Check structure
        if "mark" not in pred:
            errors.append(f"MISSING 'mark' key: {pred_path}")
            continue
        if "x" not in pred["mark"] or "y" not in pred["mark"]:
            errors.append(f"MISSING x/y in mark: {pred_path}")
            continue
        if "verified_shape" not in pred:
            errors.append(f"MISSING 'verified_shape': {pred_path}")
            continue

        # Check shape is valid
        shape = pred["verified_shape"]
        if shape not in VALID_SHAPES:
            errors.append(f"INVALID shape '{shape}': {pred_path}")
        else:
            shape_counts[shape] += 1

        # Check coordinates are numbers
        x, y = pred["mark"]["x"], pred["mark"]["y"]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            errors.append(f"NON-NUMERIC coordinates: {pred_path}")

        # Check coordinates are positive
        if x < 0 or y < 0:
            errors.append(f"NEGATIVE coordinates ({x:.1f}, {y:.1f}): {pred_path}")

    print(f"\n--- Coverage ---")
    print(f"Images in test set but missing from predictions: {len(missing_from_preds)}")
    for p in missing_from_preds[:5]:
        print(f"  MISSING: {p}")

    print(f"Predictions with no matching test image: {len(extra_in_preds)}")
    for p in extra_in_preds[:5]:
        print(f"  EXTRA: {p}")

    print(f"\n--- Shape distribution in predictions ---")
    for shape in ["Cross", "Square", "L-Shape"]:
        print(f"  {shape}: {shape_counts[shape]}")

    print(f"\n--- Format errors ---")
    if errors:
        for e in errors:
            print(f"  {e}")
    else:
        print("  None — all predictions are correctly formatted!")

    print(f"\n--- Sample predictions (first 3) ---")
    for i, (path, pred) in enumerate(list(pred_keys.items())[:3]):
        print(f"  {path}")
        print(f"    x={pred['mark']['x']:.2f}, y={pred['mark']['y']:.2f}, shape={pred['verified_shape']}")

    if not errors and not missing_from_preds:
        print("\n✓ predictions.json looks good — ready to submit!")
    else:
        print("\n✗ Issues found — review errors above before submitting.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="Path to predictions.json")
    p.add_argument("--test_root", required=True, help="Path to test_dataset root")
    args = p.parse_args()
    main(args)