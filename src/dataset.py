import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Canonical class mapping. Source data uses "L-Shape"; the assignment spec
# mentions "L-Shaped" — both are accepted on input and normalized to the
# value actually present in the labels JSON ("L-Shape") for output.
SHAPE_MAP = {"Cross": 0, "Square": 1, "L-Shape": 2, "L-Shaped": 2}
IDX_TO_SHAPE = {0: "Cross", 1: "Square", 2: "L-Shape"}

# Fixed network input size. Images are resized (preserving aspect ratio via
# LongestMaxSize) and padded to this square size.
IMG_SIZE = 512


def get_train_transforms():
    return A.Compose(
        [
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.7),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.ISONoise(p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_val_transforms():
    return A.Compose(
        [
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class GCPDataset(Dataset):
    """
    samples: list of (rel_path, label_dict) where label_dict has
        {"mark": {"x": float, "y": float}, "verified_shape": str}
    img_root: root directory the rel_path is relative to
    """

    def __init__(self, samples, img_root, transforms):
        self.samples = samples
        self.img_root = img_root
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.img_root, rel_path)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x, y = label["mark"]["x"], label["mark"]["y"]
        shape_label = SHAPE_MAP[label["verified_shape"]]

        transformed = self.transforms(image=img, keypoints=[(x, y)])
        img_t = transformed["image"]
        kp = transformed["keypoints"][0]

        # Normalize keypoint to [0, 1] in IMG_SIZE x IMG_SIZE padded space.
        kx = kp[0] / IMG_SIZE
        ky = kp[1] / IMG_SIZE

        return (
            img_t,
            torch.tensor([kx, ky], dtype=torch.float32),
            torch.tensor(shape_label, dtype=torch.long),
        )