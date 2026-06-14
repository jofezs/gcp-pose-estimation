# Aerial GCP Pose Estimation

A multi-task CNN that, given an aerial image crop, predicts:
1. The pixel `(x, y)` location of the GCP marker's center.
2. The marker's shape: `Cross`, `Square`, or `L-Shape`.

## 1. Architecture

**GCPNet** (`src/model.py`): a single ResNet-34 backbone (ImageNet-pretrained)
feeding two small heads from a shared 512-d feature vector:

- **Keypoint head**: 2 outputs, `Sigmoid`-bounded to `[0, 1]`, representing
  the normalized `(x, y)` location of the marker center in the resized/padded
  512x512 input.
- **Classification head**: 3-class logits over `{Cross, Square, L-Shape}`.

**Rationale**:
- A shared backbone is justified because the two tasks are not independent —
  the visual texture/edges that identify "this is a Cross marker" are the
  same features that localize its center. Sharing the backbone is more
  sample-efficient than training two separate networks and keeps the model
  small and easy to deploy (single forward pass, single checkpoint).
- ResNet-34 is a deliberate middle ground: deep enough to learn the relevant
  texture/edge patterns of painted ground markers against varied terrain
  (mud, gravel, concrete, vegetation), but light enough to train quickly and
  run inference on a single GPU/CPU at reasonable speed. Swapping the
  backbone is a one-line change (`--backbone resnet18/resnet50`) if more
  capacity or more speed is needed.
- Direct keypoint regression (rather than heatmap-based pose estimation) was
  chosen because there is exactly **one** keypoint per image and the marker
  occupies a small, well-defined region — a heatmap head adds complexity and
  compute without a clear benefit at this scale, and a regressed coordinate
  is simpler to post-process and evaluate against the PCK metric.

## 2. Training Strategy

### Input pipeline (`src/dataset.py`)
- `LongestMaxSize(512)` + `PadIfNeeded(512, 512)`: preserves aspect ratio
  (critical, since the dataset contains images at multiple native
  resolutions — see "Challenges" below) and avoids distorting the marker's
  shape, which would corrupt the shape-classification signal.
- All geometric augmentations use Albumentations' `KeypointParams` so the
  `(x, y)` label is transformed consistently with the image.

### Augmentations
- `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`: aerial nadir imagery has
  no canonical "up", so these are safe and meaningfully increase effective
  dataset size/orientation diversity. Critically, the `L-Shape` class is
  *not* rotation-invariant in identity (an L rotated 180° is still an "L"
  shape class, just oriented differently) — since the label is the shape
  *category*, not orientation, rotation augmentation is valid here.
- `ColorJitter`, `RandomBrightnessContrast`, `ISONoise`, `GaussianBlur`:
  drone imagery varies hugely in exposure, white balance, and JPEG
  compression artifacts across different surveys/cameras/lighting
  conditions — these augmentations target that domain-shift problem
  directly.

### Loss
- **Keypoint loss**: `SmoothL1Loss` (Huber) on normalized `(x, y)` —
  more robust to occasional annotation outliers than plain MSE, while still
  being smooth near zero (unlike pure L1).
- **Classification loss**: `CrossEntropyLoss` with **inverse-frequency class
  weights** computed from the training manifest, to counter class imbalance
  between `Cross` / `Square` / `L-Shape`.
- **Combined loss**: `kp_weight * loss_kp + loss_cls`, with `kp_weight=100`
  by default (since the normalized-coordinate SmoothL1 loss is on the order
  of 1e-3 to 1e-2 while cross-entropy is on the order of 1.0; the weight
  brings the two gradients to comparable magnitude — tune via
  `--kp_weight` if one task dominates training).

### Optimization
- `AdamW`, cosine LR schedule, ImageNet-pretrained backbone (transfer
  learning is important here given the dataset is modest in size relative to
  a from-scratch CNN).

## 3. Dataset Challenges & Mitigations

This dataset is explicitly *not* pre-sanitized, and the EDA script
(`src/eda.py`) should be run first against the real data. From the sample
provided, the following issues are already evident and handled:

1. **Inconsistent native resolution / aspect ratio.** Some annotation
   coordinates (e.g. `x=3272`, `y=2445`) exceed the 2048x1365 figure quoted
   in the spec, implying images from different drones/surveys are at
   different native resolutions (e.g. 4000x2250, 4056x3040, etc.) and/or
   different aspect ratios. **Mitigation**: the pipeline never assumes a
   fixed source resolution — image dimensions are read per-file with OpenCV,
   and `LongestMaxSize` + `PadIfNeeded` normalizes any input size to a
   512x512 square while preserving aspect ratio. Inference inverts this
   exact transform (accounting for scale + padding offset) to map predicted
   coordinates back to the original image's pixel space.

2. **Inconsistent label spelling.** The spec text says `"L-Shaped"` but the
   actual labels JSON uses `"L-Shape"`. **Mitigation**:
   `build_manifest.py` normalizes both spellings to a single canonical value
   (`"L-Shape"`, matching what's observed in the real data) and reports any
   *other* unrecognized shape strings instead of silently dropping/crashing.

3. **Highly correlated images within a survey/GCP folder.** Each
   `project/survey/gcp_id/` folder likely contains multiple photos of the
   *same physical marker* from slightly different drone positions — these
   are not independent samples. **Mitigation**: `build_manifest.py` splits
   train/val by **top-level project folder**, not by individual image, so
   validation never sees images from a project/survey the model trained on.
   This gives a much more honest estimate of generalization to the held-out
   test set (which contains entirely new projects).

4. **Class imbalance** across `Cross` / `Square` / `L-Shape`. **Mitigation**:
   inverse-frequency class weights in the classification loss
   (`compute_class_weights` in `train.py`). The EDA script reports the
   actual distribution so this can be sanity-checked / tuned further.

5. **Missing/unreadable files referenced in the labels JSON, or images on
   disk with no annotation.** **Mitigation**: `build_manifest.py` checks file
   existence and readability for every JSON entry, logs how many are missing
   (and which ones, for the first few), and only includes verifiably present
   images in the train/val manifests.

6. **Coordinates landing outside image bounds** (possible annotation
   errors). `eda.py` explicitly counts these so they can be inspected and,
   if needed, excluded or corrected before training.

## 4. Reproducing `predictions.json`

```bash
cd src

# 1. EDA (run first against the real dataset to confirm assumptions above)
python eda.py --data_root /path/to/train_dataset \
               --labels_json /path/to/train_dataset/curated_gcp_marks.json

# 2. Build train/val manifests (leakage-free split by project)
python build_manifest.py --data_root /path/to/train_dataset \
                          --labels_json /path/to/train_dataset/curated_gcp_marks.json \
                          --out_dir ../manifests

# 3. Train
python train.py --data_root /path/to/train_dataset \
                 --manifest_dir ../manifests \
                 --out_dir ../checkpoints \
                 --epochs 30 --batch_size 16

# 4. Inference on the unlabelled test set -> predictions.json
python infer.py --data_root /path/to/test_dataset \
                 --weights ../checkpoints/best_model.pt \
                 --out_file ../predictions.json
```

## 5. Model Weights

Trained weights (`best_model.pt`) should be uploaded to cloud storage (e.g.
a Google Drive / S3 link) and referenced here once training has been run
against the full dataset — the code in this repo is the deliverable;
`infer.py --weights <path>` is the single entry point needed to regenerate
`predictions.json` from any checkpoint produced by `train.py`.

## 6. Repository Structure

```
gcp_pose_estimation/
├── README.md
├── requirements.txt
├── manifests/              # generated by build_manifest.py
├── checkpoints/            # generated by train.py
└── src/
    ├── eda.py
    ├── build_manifest.py
    ├── dataset.py
    ├── model.py
    ├── train.py
    └── infer.py
```
