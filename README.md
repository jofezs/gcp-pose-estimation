# Aerial GCP Pose Estimation

A multi-task deep learning pipeline that takes an aerial drone image as input and simultaneously predicts:
1. The exact pixel `(x, y)` coordinates of the GCP marker's center
2. The shape of the marker: `Cross`, `Square`, or `L-Shape`

---

## Table of Contents
- [Architecture](#architecture)
- [Training Strategy](#training-strategy)
- [Dataset Challenges](#dataset-challenges)
- [Results](#results)
- [Model Weights](#model-weights)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)

---

## Architecture

**Model: GCPNet** (`src/model.py`)

A shared ResNet-34 backbone with two task-specific heads:

```
Input Image (512x512)
       │
  ResNet-34 Backbone (ImageNet pretrained)
       │
  AdaptiveAvgPool → 512-d feature vector → Dropout(0.3)
       │                    │
  Keypoint Head        Classification Head
  Linear → ReLU        Linear → ReLU
  Linear → Sigmoid     Linear → Logits
  (x, y) ∈ [0,1]      (Cross / Square / L-Shape)
```

A single ResNet-34 backbone feeds both heads. This works well because finding the marker center and identifying its shape rely on the same visual features — there is no benefit in running two separate models. A single forward pass at inference time gives both results.

ResNet-34 was chosen because it is pre-trained on ImageNet (strong starting point), not too large to overfit on ~850 images, and fast enough to train and run inference without a high-end GPU.

---

## Training Strategy

### Input Pipeline (`src/dataset.py`)

All images are resized and padded to a fixed **512×512** square:
- `LongestMaxSize(512)` — scales the longest side to 512, preserving aspect ratio (critical since the dataset contains two different native resolutions: 4096×2730 and 4096×3068)
- `PadIfNeeded(512, 512)` — zero-pads the shorter side to make a square

All geometric transforms use Albumentations' `KeypointParams` so the `(x, y)` label is transformed consistently with the image. At inference time, the padding and scale are inverted to map predictions back to the original image's pixel coordinates.

### Augmentations

| Augmentation | Reason |
|---|---|
| `HorizontalFlip`, `VerticalFlip`, `RandomRotate90` | Aerial nadir imagery has no canonical orientation — these are safe and double/quadruple effective dataset size |
| `ColorJitter`, `RandomBrightnessContrast` | Drone imagery varies hugely in exposure and white balance across surveys/cameras/lighting |
| `GaussianBlur`, `ISONoise` | Targets JPEG compression artifacts and sensor noise variation across different drone models |

### Loss Functions

**Keypoint loss:** `SmoothL1Loss` (Huber) on normalized `(x, y)` coordinates — more robust to occasional annotation outliers than MSE, while remaining smooth near zero unlike pure L1.

**Classification loss:** `CrossEntropyLoss` with inverse-frequency class weights computed from the training set to counter class imbalance:
- Cross: 177 samples → higher weight
- Square: 328 samples → medium weight  
- L-Shape: 491 samples → lower weight

**Combined loss:** `total = kp_weight × loss_kp + loss_cls`

A `kp_weight` of 100 is used to bring the small normalized-coordinate SmoothL1 loss (order of 1e-3) to a comparable gradient magnitude as the cross-entropy loss (order of 1.0).

### Optimization
- Optimizer: `AdamW` with weight decay 1e-4
- Schedule: Cosine annealing LR over 30 epochs
- Batch size: 16
- Best checkpoint saved automatically based on lowest validation loss

---

## Dataset Challenges

This dataset reflects real production conditions and required careful handling:

**1. Inconsistent image resolutions**
Images come from different drone models at two native resolutions (4096×2730 and 4096×3068). Some annotation coordinates also exceed the 2048×1365 figure quoted in the spec. The pipeline reads actual image dimensions per file with OpenCV and uses aspect-ratio-preserving resize + pad rather than assuming any fixed size.

**2. Inconsistent shape label spelling**
The JSON uses `"L-Shape"` while the spec document says `"L-Shaped"`. The manifest builder normalizes both spellings to a single canonical value and reports any other unrecognized strings rather than silently crashing or dropping samples.

**3. Missing shape labels**
4 out of 1000 entries had `verified_shape: null`. These were detected during EDA and excluded from training/validation with explicit logging.

**4. Class imbalance**
L-Shape (491) significantly outnumbers Cross (177) and Square (328). Mitigated with inverse-frequency class weights in the classification loss.

**5. Correlated images within a GCP folder**
Each `project/survey/gcp_id/` folder contains multiple photos of the same physical marker from slightly different drone positions — these are not independent samples. A naive random split would leak these into both train and val, inflating validation metrics. The split was done at the **GCP folder level**, keeping each `gcp_id` folder entirely in train or entirely in val, with stratified sampling per class to ensure all 3 shapes appear in both splits.

**6. Only 11 top-level projects**
A project-level split (the ideal for preventing domain leakage) resulted in val having zero Cross samples when large single-class projects (like Vedanta GOA Bicholim, 254 images, all L-Shape) ended up in val. The GCP-level stratified split was the pragmatic fix that balanced leakage prevention with class coverage.

---

## Results

Training on 850 samples, validating on 146 samples (GCP-level stratified split):

| Metric | Value |
|---|---|
| Val classification accuracy | 97.9 – 98.6% |
| Val keypoint loss (normalized) | ~0.020 |
| Approx. pixel error at 512px input | ~10px |
| Approx. pixel error at original resolution | ~80px |

The keypoint error of ~80px on 4096px images should comfortably satisfy PCK@50px thresholds for most test images, with room for improvement via test-time augmentation or a larger backbone.

---

## Model Weights

Download `best_model.pt` from:

> https://drive.google.com/file/d/1OeC0mFP6EJ91g7TJU1ZdWll5bBQuGvjT/view?usp=sharing

Place the downloaded file at `checkpoints/best_model.pt` before running inference.

---

## How to Run

### Reproducing predictions.json (Quick Start)

If you just want to run inference and reproduce `predictions.json`:

```bash
# 1. Clone and install
git clone https://github.com/jofezs/gcp-pose-estimation.git
cd gcp_pose_estimation
pip install -r requirements.txt

# 2. Download best_model.pt from the link above and place it at:
#    checkpoints/best_model.pt

# 3. Run inference (replace the dataset path with your path to dataset)
cd src
python infer.py \
    --data_root ../test_dataset \
    --weights ../checkpoints/best_model.pt \
    --out_file ../predictions.json
```

That's all that's needed to reproduce `predictions.json`. The sections below describe the full pipeline including EDA, dataset splitting, and training from scratch.

---

### Path Variables

Before running any command, set these paths to match where your data lives on your machine. Every `--flag` below corresponds to one of these:

| Variable | Flag | Description | Example value |
|---|---|---|---|
| `TRAIN_ROOT` | `--data_root` | Root folder of the training dataset | `../train_dataset` |
| `TEST_ROOT` | `--data_root` | Root folder of the test dataset | `../test_dataset` |
| `LABELS_JSON` | `--labels_json` | Path to the annotations file inside the training dataset | `../train_dataset/gcp_marks.json` |
| `MANIFEST_DIR` | `--manifest_dir` / `--out_dir` | Where to save/read the train & val split files | `../manifests` |
| `CHECKPOINT_DIR` | `--out_dir` / `--weights` | Where to save/read model weights | `../checkpoints` |
| `PREDICTIONS_FILE` | `--out_file` / `--predictions` | Output path for predictions JSON | `../predictions.json` |

The example values below assume you run all commands from inside the `src/` folder (`cd src`) with `train_dataset` and `test_dataset` sitting one level up. Adjust the paths if your folder layout differs.

---

### 1. Setup
```bash
git clone https://github.com/jofezs/gcp-pose-estimation.git
cd gcp_pose_estimation
python -m venv venv
# Windows:
venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
cd src
```

### 2. EDA (recommended first step)
Inspects the dataset — reports image resolutions, class distribution, missing files, and project breakdown.
```bash
# --data_root   → TRAIN_ROOT  (path to training images)
# --labels_json → LABELS_JSON (path to annotations file)
python eda.py \
    --data_root ../train_dataset \
    --labels_json ../train_dataset/gcp_marks.json
```

### 3. Build manifests
Creates `train_manifest.json` and `val_manifest.json` — a leakage-free split at the GCP folder level, stratified by shape class.
```bash
# --data_root   → TRAIN_ROOT    (path to training images)
# --labels_json → LABELS_JSON   (path to annotations file)
# --out_dir     → MANIFEST_DIR  (where to save the split files)
python build_manifest.py \
    --data_root ../train_dataset \
    --labels_json ../train_dataset/gcp_marks.json \
    --out_dir ../manifests
```

### 4. Train
Trains GCPNet and saves the best checkpoint (lowest val loss) to `CHECKPOINT_DIR/best_model.pt`.
```bash
# --data_root    → TRAIN_ROOT     (path to training images)
# --manifest_dir → MANIFEST_DIR   (folder containing the split files from step 3)
# --out_dir      → CHECKPOINT_DIR (where to save model weights)
python train.py \
    --data_root ../train_dataset \
    --manifest_dir ../manifests \
    --out_dir ../checkpoints \
    --epochs 30 \
    --batch_size 16
```

Optional flags:
- `--backbone resnet18` — faster, use if training on CPU
- `--batch_size 8` — reduce if running out of RAM
- `--workers 0` — set to 0 on Windows if multiprocessing errors occur
- `--lr 1e-4` — learning rate (default)
- `--kp_weight 100` — scale factor balancing keypoint vs classification loss

### 5. Inference → predictions.json
Runs the trained model over every image in the test dataset and writes `PREDICTIONS_FILE`.
```bash
# --data_root → TEST_ROOT        (path to test images)
# --weights   → CHECKPOINT_DIR/best_model.pt (trained model weights)
# --out_file  → PREDICTIONS_FILE (output path for predictions JSON)
python infer.py \
    --data_root ../test_dataset \
    --weights ../checkpoints/best_model.pt \
    --out_file ../predictions.json
```

### 6. Validate predictions format
Checks that every test image has a prediction, coordinates are valid, and shape labels are correctly named.
```bash
# --predictions → PREDICTIONS_FILE (the JSON produced in step 5)
# --test_root   → TEST_ROOT        (path to test images, for coverage check)
python validate_predictions.py \
    --predictions ../predictions.json \
    --test_root ../test_dataset
```

---


## Repository Structure

```
gcp_pose_estimation/
├── README.md
├── requirements.txt
├── predictions.json
├── manifests/
│   ├── train_manifest.json
│   └── val_manifest.json
├── checkpoints/
│   └── best_model.pt
└── src/
    ├── eda.py                  # Exploratory data analysis
    ├── build_manifest.py       # Dataset splitting and label normalization
    ├── dataset.py              # PyTorch Dataset + augmentation pipeline
    ├── model.py                # GCPNet architecture
    ├── train.py                # Training loop
    ├── infer.py                # Inference → predictions.json
    └── validate_predictions.py # Format and coverage validation
```