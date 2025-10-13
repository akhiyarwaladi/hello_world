#!/usr/bin/env python3
"""
Verify MD_2019 Bbox Source: Mask vs Hardcoded

Checks how many bboxes were extracted from masks vs hardcoded estimates.
This script analyzes the YOLO-format labels to determine bbox size distribution.

Author: Investigation Team
Date: 2025-10-14
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

def analyze_bbox_sizes():
    """Analyze bbox sizes in MD_2019 YOLO format"""

    labels_dir = Path("data/raw/md_2019/labels")

    if not labels_dir.exists():
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return

    # Collect bbox sizes per class
    bbox_sizes = {0: [], 1: [], 2: []}  # ring, schizont, trophozoite
    stage_names = {0: 'ring', 1: 'schizont', 2: 'trophozoite'}

    # Expected hardcoded sizes (from script)
    hardcoded_sizes = {
        0: (40, 36),   # Ring
        1: (91, 96),   # Schizont
        2: (71, 70)    # Trophozoite
    }

    total_annotations = 0
    hardcoded_count = {0: 0, 1: 0, 2: 0}

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])

                    # Store normalized sizes
                    bbox_sizes[class_id].append((width_norm, height_norm))
                    total_annotations += 1

                    # Detect if this bbox is likely hardcoded
                    # Hardcoded sizes would have very consistent normalized values
                    # since they're always the same pixel size

                except (ValueError, IndexError, KeyError):
                    continue

    print("="*80)
    print("MD_2019 BBOX SIZE ANALYSIS")
    print("="*80)
    print(f"\nTotal annotations: {total_annotations}")

    # Analyze size distribution per class
    for class_id in [0, 1, 2]:
        sizes = bbox_sizes[class_id]
        stage_name = stage_names[class_id]

        if not sizes:
            print(f"\n{stage_name.upper()}: No annotations found")
            continue

        widths = [w for w, h in sizes]
        heights = [h for w, h in sizes]

        width_mean = np.mean(widths)
        width_std = np.std(widths)
        width_cv = (width_std / width_mean * 100) if width_mean > 0 else 0

        height_mean = np.mean(heights)
        height_std = np.std(heights)
        height_cv = (height_std / height_mean * 100) if height_mean > 0 else 0

        # Count unique sizes
        unique_widths = len(set(np.round(widths, 6)))
        unique_heights = len(set(np.round(heights, 6)))

        print(f"\n{stage_name.upper()} (class {class_id}):")
        print(f"  Samples: {len(sizes)}")
        print(f"  Width (normalized):")
        print(f"    Mean: {width_mean:.6f}")
        print(f"    Std Dev: {width_std:.6f}")
        print(f"    CV: {width_cv:.2f}%")
        print(f"    Unique values: {unique_widths}")
        print(f"  Height (normalized):")
        print(f"    Mean: {height_mean:.6f}")
        print(f"    Std Dev: {height_std:.6f}")
        print(f"    CV: {height_cv:.2f}%")
        print(f"    Unique values: {unique_heights}")

        # Show most common sizes
        width_counter = Counter([round(w, 6) for w in widths])
        height_counter = Counter([round(h, 6) for h in heights])

        print(f"  Most common widths (top 5):")
        for width, count in width_counter.most_common(5):
            pct = count / len(widths) * 100
            print(f"    {width:.6f}: {count} ({pct:.1f}%)")

        print(f"  Most common heights (top 5):")
        for height, count in height_counter.most_common(5):
            pct = count / len(heights) * 100
            print(f"    {height:.6f}: {count} ({pct:.1f}%)")

        # Check if sizes are suspiciously uniform
        if width_cv < 5 and height_cv < 5:
            print(f"  ⚠️  WARNING: Extremely uniform sizes (CV < 5%) - likely hardcoded!")

        if unique_widths < 10 and unique_heights < 10:
            print(f"  ⚠️  WARNING: Very few unique sizes - likely hardcoded!")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("\nIf CV < 5% and unique values < 10, the bboxes are likely hardcoded.")
    print("This explains the 99.82% accuracy - model learns size patterns, not morphology.")
    print("\nPaper original reported 82.7% accuracy with random forest + 112 features.")
    print("Our 99.82% with CNN is unrealistic and indicates dataset preprocessing error.")

if __name__ == "__main__":
    analyze_bbox_sizes()
