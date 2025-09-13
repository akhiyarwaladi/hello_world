#!/usr/bin/env python3
"""
Draw YOLO GT boxes on a few images to visually verify crops/labels.

Usage:
  python scripts/11_debug_draw_boxes.py --split train --limit 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2

SPLIT_ROOT = Path('data/detection/split')
LABELS_DIR = Path('data/detection/labels')
OUT_DIR = Path('results/debug_boxes')


def read_list(split: str) -> list[Path]:
    p = SPLIT_ROOT / f'{split}.txt'
    if not p.exists():
        raise SystemExit(f'Not found: {p}. Run 09_train_detection.py first to create list-based split.')
    return [Path(x.strip()) for x in p.read_text().splitlines() if x.strip()]


def yolo_to_xyxy(line: str, w: int, h: int) -> tuple[int,int,int,int]:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError('bad yolo line')
    _, cx, cy, bw, bh = parts
    cx = float(cx) * w
    cy = float(cy) * h
    bw = float(bw) * w
    bh = float(bh) * h
    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train','val','test'])
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imgs = read_list(args.split)[: args.limit]

    for img_path in imgs:
        im = cv2.imread(str(img_path))
        if im is None:
            print('skip unreadable:', img_path)
            continue
        h, w = im.shape[:2]
        lab_path = LABELS_DIR / f'{img_path.stem}.txt'
        if not lab_path.exists():
            print('no label for', img_path)
            continue
        for ln in lab_path.read_text().splitlines():
            try:
                x1, y1, x2, y2 = yolo_to_xyxy(ln, w, h)
            except Exception:
                continue
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 3)

        out = OUT_DIR / f'{img_path.stem}_debug.jpg'
        cv2.imwrite(str(out), im)
        print('wrote', out)


if __name__ == '__main__':
    main()

