#!/usr/bin/env python3
"""
Copy Best Model Visualizations to visualization_outputs

Copies only YOLO11 (detection) and EfficientNet-B1 (classification) visualizations
from experiment folder to visualization_outputs for easy access.

Total: ~1 GB (vs 3.5 GB for all models)
"""

import shutil
from pathlib import Path
import pandas as pd

def main():
    # Paths
    experiment_root = Path("results/optA_20251207_233941/experiments")
    output_root = Path("visualization_outputs/full")

    print("=" * 80)
    print("COPYING BEST MODEL VISUALIZATIONS")
    print("=" * 80)
    print(f"From: {experiment_root}")
    print(f"To: {output_root}")
    print()

    # Best models
    best_detection = "yolo11"
    best_classification = "efficientnet_b1_focal"

    datasets = [
        "iml_lifecycle",
        "mp_idb_species",
        "mp_idb_stages",
        "md_2019_stages"
    ]

    # Clean output folder
    if output_root.exists():
        print(f"🗑️  Cleaning {output_root}...")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "detection").mkdir(exist_ok=True)
    (output_root / "classification").mkdir(exist_ok=True)

    total_copied = 0

    # Copy detection visualizations (YOLO11)
    print(f"\n[1/2] COPYING DETECTION VISUALIZATIONS ({best_detection})")
    print("-" * 80)

    for dataset in datasets:
        src_folder = experiment_root / f"experiment_{dataset}" / "visualizations" / f"pred_detection_{best_detection}"
        dest_folder = output_root / "detection" / f"{best_detection}_{dataset}"

        if not src_folder.exists():
            print(f"  ⚠️  {dataset}: Source not found - {src_folder}")
            continue

        # Copy all PNG files
        pngs = list(src_folder.glob("*.png"))
        dest_folder.mkdir(parents=True, exist_ok=True)

        for png in pngs:
            shutil.copy2(png, dest_folder / png.name)

        total_copied += len(pngs)
        print(f"  ✓ {dataset}: Copied {len(pngs)} images → {dest_folder.name}/")

    # Copy classification visualizations (EfficientNet-B1)
    print(f"\n[2/2] COPYING CLASSIFICATION VISUALIZATIONS ({best_classification})")
    print("-" * 80)

    for dataset in datasets:
        src_folder = experiment_root / f"experiment_{dataset}" / "visualizations" / f"pred_classification_{best_classification}"
        dest_folder = output_root / "classification" / f"{best_classification}_{dataset}"

        if not src_folder.exists():
            print(f"  ⚠️  {dataset}: Source not found - {src_folder}")
            continue

        # Copy all PNG files
        pngs = list(src_folder.glob("*.png"))
        dest_folder.mkdir(parents=True, exist_ok=True)

        for png in pngs:
            shutil.copy2(png, dest_folder / png.name)

        total_copied += len(pngs)
        print(f"  ✓ {dataset}: Copied {len(pngs)} images → {dest_folder.name}/")

    print()
    print("=" * 80)
    print(f"✅ COPY COMPLETE!")
    print("=" * 80)
    print(f"Total files copied: {total_copied}")
    print(f"Output location: {output_root}")
    print()
    print("Structure:")
    print(f"  visualization_outputs/full/")
    print(f"    ├── detection/")
    for dataset in datasets:
        print(f"    │   ├── {best_detection}_{dataset}/")
    print(f"    └── classification/")
    for dataset in datasets:
        print(f"        ├── {best_classification}_{dataset}/")
    print()

if __name__ == "__main__":
    main()
