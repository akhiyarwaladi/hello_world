#!/usr/bin/env python3
"""
Crop parasite regions into a species-labeled classification dataset.

Default mode uses ground-truth YOLO labels from data/detection/split/labels
and species mapping from MP-IDB CSVs to create:
  data/classification/{train,val,test}/{falciparum,vivax,malariae,ovale}/*.jpg

Optionally, you can crop from model predictions if you have a trained detector.

Usage examples:
  python scripts/10_crop_detections.py                            # use GT labels
  python scripts/10_crop_detections.py --pred-weights results/detection/yolo11n_mpidb/weights/best.pt
  python scripts/10_crop_detections.py --padding 1.3 --min-size 12
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # inference only if predictions mode
    YOLO = None  # type: ignore


MPIDB_ROOT = Path("data/raw/mp_idb")
DET_SPLIT_ROOT = Path("data/detection/split")
DET_IMAGES_DIR = Path("data/detection/images")
DET_LABELS_DIR = Path("data/detection/labels")
DEFAULT_OUT_ROOT = Path("data/classification_crops")

SPECIES_DIRS = ["Falciparum", "Vivax", "Malariae", "Ovale"]
SPECIES_CANON = {
    "Falciparum": "falciparum",
    "Vivax": "vivax",
    "Malariae": "malariae",
    "Ovale": "ovale",
}


def _load_species_mapping(mpidb_root: Path) -> Dict[str, str]:
    """Build mapping from base image stem -> species name (canonical lower-case)."""
    mapping: Dict[str, str] = {}
    for sp in SPECIES_DIRS:
        csv_path = mpidb_root / sp / f"mp-idb-{sp.lower()}.csv"
        if not csv_path.exists():
            # Try abspath variant if present
            alt = mpidb_root / sp / f"mp-idb-{sp.lower()}-abspath.csv"
            if alt.exists():
                csv_path = alt
            else:
                continue
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = Path(row.get("filename", "")).name
                if not fn:
                    continue
                mapping[Path(fn).stem] = SPECIES_CANON[sp]
    if not mapping:
        raise RuntimeError("Could not build species mapping from MP-IDB CSVs.")
    return mapping


def _yolo_to_xyxy(b: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    cx, cy, bw, bh = b
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2


def _pad_and_clip(x1, y1, x2, y2, pad: float, w: int, h: int) -> Tuple[int, int, int, int]:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * pad
    bh = (y2 - y1) * pad
    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w - 1, int(cx + bw / 2))
    ny2 = min(h - 1, int(cy + bh / 2))
    return nx1, ny1, nx2, ny2


def _ensure_out_dirs(out_root: Path):
    for split in ["train", "val", "test"]:
        for sp in SPECIES_CANON.values():
            (out_root / split / sp).mkdir(parents=True, exist_ok=True)


def _read_split_list(split: str) -> List[Path]:
    list_path = DET_SPLIT_ROOT / f"{split}.txt"
    if not list_path.exists():
        return []
    imgs: List[Path] = []
    with open(list_path, "r") as f:
        for ln in f:
            p = Path(ln.strip())
            if p.exists():
                imgs.append(p)
    return imgs


def _iter_gt_label_files() -> List[Tuple[str, Path, Path]]:
    """Return list of (split, img_path, lbl_path) using list-based split to avoid duplication."""
    items: List[Tuple[str, Path, Path]] = []
    for split in ["train", "val", "test"]:
        imgs = _read_split_list(split)
        for img in imgs:
            lbl = DET_LABELS_DIR / f"{img.stem}.txt"
            if lbl.exists():
                items.append((split, img, lbl))
    if not items:
        raise RuntimeError("No GT split lists found. Run 09_train_detection.py to create train.txt/val.txt/test.txt.")
    return items


def crop_from_gt(padding: float, min_size: int, output_size: int | None, out_root: Path) -> None:
    mapping = _load_species_mapping(MPIDB_ROOT)
    _ensure_out_dirs(out_root)

    index: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    items = _iter_gt_label_files()
    for split, img_path, lbl_path in items:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(lbl_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        for i, ln in enumerate(lines):
            parts = ln.split()
            if len(parts) != 5:
                continue
            # class_id = int(parts[0])  # always 0 (parasite)
            cx, cy, bw, bh = map(float, parts[1:])
            x1, y1, x2, y2 = _yolo_to_xyxy((cx, cy, bw, bh), w, h)
            x1, y1, x2, y2 = _pad_and_clip(x1, y1, x2, y2, padding, w, h)
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                continue

            crop = img[y1:y2, x1:x2]
            if output_size is not None and output_size > 0:
                crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
            species = mapping.get(img_path.stem)
            if species is None:
                # fall back to unknown; skip to keep dataset clean
                continue

            out_dir = out_root / split / species
            out_name = f"{img_path.stem}_{i}.jpg"
            out_path = out_dir / out_name
            if not out_path.exists():
                cv2.imwrite(str(out_path), crop)

            index[split].append({
                "src": str(img_path),
                "label": species,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "out": str(out_path),
            })

    # Save an index for reproducibility
    (out_root).mkdir(parents=True, exist_ok=True)
    with open(out_root / "crops_index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Basic summary
    counts = {split: len(index[split]) for split in index}
    print("\n✅ Cropping completed (ground-truth).")
    print(f"Saved crops per split: {counts}")
    print(f"Output directory: {out_root}")


def crop_from_predictions(weights: str, padding: float, min_size: int, conf: float, output_size: int | None, out_root: Path) -> None:
    if YOLO is None:
        raise RuntimeError("Ultralytics not available. Install per requirements.txt")
    model = YOLO(weights)
    _ensure_out_dirs(out_root)

    # We'll run predictions over the split images to keep the species mapping
    mapping = _load_species_mapping(MPIDB_ROOT)
    index: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        imgs = _read_split_list(split)
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            preds = model.predict(str(img_path), conf=conf, verbose=False)
            if not preds:
                continue
            boxes = preds[0].boxes.xyxy.cpu().numpy() if hasattr(preds[0].boxes, 'xyxy') else []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1, y1, x2, y2 = _pad_and_clip(x1, y1, x2, y2, padding, w, h)
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    continue
                crop = img[y1:y2, x1:x2]
                species = mapping.get(img_path.stem)
                if species is None:
                    continue
                out_dir = out_root / split / species
                out_name = f"{img_path.stem}_{i}.jpg"
                out_path = out_dir / out_name
                if output_size is not None and output_size > 0:
                    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
                if not out_path.exists():
                    cv2.imwrite(str(out_path), crop)
                index[split].append({
                    "src": str(img_path),
                    "label": species,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "out": str(out_path),
                })

    with open(out_root / "crops_index.json", "w") as f:
        json.dump(index, f, indent=2)
    counts = {split: len(index[split]) for split in index}
    print("\n✅ Cropping completed (predictions).")
    print(f"Saved crops per split: {counts}")
    print(f"Output directory: {out_root}")


def main():
    ap = argparse.ArgumentParser(description="Crop parasite regions into species-labeled classification dataset")
    ap.add_argument("--use-gt-labels", action="store_true", help="Use ground-truth YOLO labels (default)")
    ap.add_argument("--pred-weights", default="", help="Use model predictions with this weights path")
    ap.add_argument("--padding", type=float, default=1.20, help="BBox padding factor (e.g., 1.2)")
    ap.add_argument("--min-size", type=int, default=10, help="Discard crops smaller than this size (px)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold if using predictions")
    ap.add_argument("--output-size", type=int, default=96, help="Optional resize of crops to NxN (saves disk)")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output root for cropped classification dataset")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if args.use_gt_labels or not args.pred_weights:
        crop_from_gt(padding=args.padding, min_size=args.min_size, output_size=args.output_size, out_root=out_root)
    else:
        crop_from_predictions(weights=args.pred_weights, padding=args.padding, min_size=args.min_size, conf=args.conf, output_size=args.output_size, out_root=out_root)

    print("\nNext step: Train the classifier on the cropped dataset using:")
    print("  python scripts/07_train_yolo_quick.py --data", out_root)


if __name__ == "__main__":
    main()
