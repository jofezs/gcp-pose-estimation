"""
Trains GCPNet on the manifests produced by build_manifest.py.

Run:
    python train.py --data_root /path/to/train_dataset \
        --manifest_dir manifests --out_dir checkpoints \
        --epochs 30 --batch_size 16
"""

import os
import json
import argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GCPDataset, get_train_transforms, get_val_transforms, SHAPE_MAP
from model import GCPNet


def compute_class_weights(samples):
    """Inverse-frequency class weights to counter class imbalance."""
    counts = Counter(SHAPE_MAP[label["verified_shape"]] for _, label in samples)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / counts.get(i, 1) for i in range(3)], dtype=torch.float32
    )
    weights = weights / weights.sum() * len(weights)
    return weights


def run_epoch(model, loader, optimizer, device, kp_loss_fn, cls_loss_fn, kp_weight, train=True):
    model.train() if train else model.eval()

    total_loss = total_kp = total_cls = 0.0
    correct = n = 0

    for imgs, kps, labels in loader:
        imgs, kps, labels = imgs.to(device), kps.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            pred_kp, pred_cls = model(imgs)
            loss_kp = kp_loss_fn(pred_kp, kps)
            loss_cls = cls_loss_fn(pred_cls, labels)
            loss = kp_weight * loss_kp + loss_cls

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_kp += loss_kp.item() * bs
        total_cls += loss_cls.item() * bs
        correct += (pred_cls.argmax(1) == labels).sum().item()
        n += bs

    return total_loss / n, total_kp / n, total_cls / n, correct / n


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(os.path.join(args.manifest_dir, "train_manifest.json")) as f:
        train_samples = json.load(f)
    with open(os.path.join(args.manifest_dir, "val_manifest.json")) as f:
        val_samples = json.load(f)

    train_ds = GCPDataset(train_samples, args.data_root, get_train_transforms())
    val_ds = GCPDataset(val_samples, args.data_root, get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
    )

    model = GCPNet(backbone=args.backbone).to(device)

    class_weights = compute_class_weights(train_samples).to(device)
    print(f"Class weights (Cross, Square, L-Shape): {class_weights.tolist()}")

    kp_loss_fn = nn.SmoothL1Loss()
    cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        tr = run_epoch(model, train_loader, optimizer, device, kp_loss_fn, cls_loss_fn, args.kp_weight, train=True)
        va = run_epoch(model, val_loader, optimizer, device, kp_loss_fn, cls_loss_fn, args.kp_weight, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train: loss={tr[0]:.4f} kp={tr[1]:.5f} cls={tr[2]:.4f} acc={tr[3]:.3f} | "
            f"val: loss={va[0]:.4f} kp={va[1]:.5f} cls={va[2]:.4f} acc={va[3]:.3f}"
        )

        if va[0] < best_val:
            best_val = va[0]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print(f"  -> new best (val_loss={va[0]:.4f}), saved best_model.pt")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "last_model.pt"))
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Path to train_dataset root")
    p.add_argument("--manifest_dir", default="manifests")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--backbone", default="resnet34")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    # kp_weight scales the (small, [0,1]-normalized) SmoothL1 keypoint loss
    # so it's on a comparable magnitude to the cross-entropy loss.
    p.add_argument("--kp_weight", type=float, default=100.0)
    args = p.parse_args()
    main(args)