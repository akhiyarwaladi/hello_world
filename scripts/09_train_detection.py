#!/usr/bin/env python3
"""
Train a YOLO detector on MP-IDB-derived bounding boxes (two-stage pipeline, stage 2).

This script expects you to have run:
  python scripts/08_parse_mpid_detection.py

It will:
  1) Create a train/val/test split under data/detection/split
  2) Write a YOLO dataset.yaml pointing to the split
  3) Train a YOLO detector (default yolo11n.pt) on the split

Usage examples:
  python scripts/09_train_detection.py                               # default settings
  python scripts/09_train_detection.py --epochs 100 --device cuda:0  # longer GPU run
  python scripts/09_train_detection.py --model yolov8n.pt            # use YOLOv8
  python scripts/09_train_detection.py --batch 32 --imgsz 768
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from ultralytics import YOLO


DEF_DATA_ROOT = Path("data/detection")
DEF_SPLIT_ROOT = DEF_DATA_ROOT / "split"


def _pick_default_model() -> str:
    """Choose a reasonable default model checkpoint available in the repo or fallback to Ultralytics hub."""
    candidates = [
        "yolo11n.pt",
        "yolov8n.pt",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return "yolo11n.pt"  # will auto-download if not present


def _gather_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for img in sorted(images_dir.glob("*.jpg")):
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def _ensure_split_dirs(root: Path):
    # For list-based splits we only need the folder to hold txt files
    root.mkdir(parents=True, exist_ok=True)


def _write_split_yaml(split_root: Path) -> Path:
    train_txt = (split_root / "train.txt").resolve()
    val_txt = (split_root / "val.txt").resolve()
    test_txt = (split_root / "test.txt").resolve()
    yaml_text = f"""# MP-IDB Parasite Detection (split, list-based)
train: {train_txt}
val: {val_txt}
test: {test_txt}

nc: 1
names:
  0: parasite
"""
    out = split_root / "dataset.yaml"
    out.write_text(yaml_text)
    return out


def create_split(data_root: Path, split_root: Path, train=0.8, val=0.1, seed=42) -> Path:
    """Create a deterministic split from data_root/images + data_root/labels into split_root.

    Returns path to the split dataset.yaml
    """
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    assert images_dir.exists() and labels_dir.exists(), "Run 08_parse_mpid_detection.py first."

    pairs = _gather_pairs(images_dir, labels_dir)
    if not pairs:
        raise RuntimeError("No image/label pairs found in data/detection. Did parsing succeed?")

    random.Random(seed).shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val :]

    _ensure_split_dirs(split_root)

    # Write list files referencing original images (no file duplication)
    def write_list(pairs: List[Tuple[Path, Path]], fname: str):
        with open(split_root / fname, "w") as f:
            for img, _ in pairs:
                f.write(str(img.resolve()) + "\n")

    write_list(train_pairs, "train.txt")
    write_list(val_pairs, "val.txt")
    write_list(test_pairs, "test.txt")

    print(f"Split created (list-based): {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
    return _write_split_yaml(split_root)


def train_detector(dataset_yaml: Path, model_path: str, epochs: int, batch: int, imgsz: int, device: str,
                   project: str, name: str):
    print("\n=== Training Configuration ===")
    print(f"data     : {dataset_yaml}")
    print(f"model    : {model_path}")
    print(f"epochs   : {epochs}")
    print(f"batch    : {batch}")
    print(f"imgsz    : {imgsz}")
    print(f"device   : {device}")
    print(f"project  : {project}")
    print(f"name     : {name}")

    model = YOLO(model_path)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        optimizer="auto",
        patience=20,
        cos_lr=True,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        cache=True,
        seed=42,
        amp=True,
    )

    print("\n✅ Detection training finished.")
    print(f"Run dir         : {Path(project) / name}")


def main():
    ap = argparse.ArgumentParser(description="Train YOLO detector on MP-IDB-derived labels")
    ap.add_argument("--data-root", default=str(DEF_DATA_ROOT), help="Parsed detection root (with images/labels)")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size")
    ap.add_argument("--device", default="cpu", help="Device string, e.g., 'cuda:0' or 'cpu'")
    ap.add_argument("--model", default=_pick_default_model(), help="Model checkpoint path or name")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio (rest goes to test)")
    ap.add_argument("--project", default="results/detection", help="Ultralytics project dir")
    ap.add_argument("--name", default="yolo11n_mpidb", help="Run name")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}. Run 08_parse_mpid_detection.py first.")

    dataset_yaml = create_split(data_root, DEF_SPLIT_ROOT, train=args.train_ratio, val=args.val_ratio)
    train_detector(dataset_yaml, args.model, args.epochs, args.batch, args.imgsz, args.device, args.project, args.name)

    print(
        "\nNext steps:\n"
        "  1) Evaluate detection results in results/detection\n"
        "  2) Crop detected regions to build species classification dataset:\n"
        "     python scripts/10_crop_detections.py --use-gt-labels\n"
        "  3) Train the classifier on the cropped dataset:\n"
        "     python scripts/07_train_yolo_quick.py"
    )


if __name__ == "__main__":
    main()
