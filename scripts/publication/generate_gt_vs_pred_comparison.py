#!/usr/bin/env python3
"""
Generate Ground Truth vs Prediction Side-by-Side Comparison Images

For 3-column paper layout, creates comparison images showing:
- Left: Ground Truth detection
- Right: Predicted detection
- Labeled with text overlay

Useful for visualizing False Negatives and False Positives.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "optA_20251016_200330" / "experiments"
OUTPUT_DIR = BASE_DIR / "luaran" / "templates" / "figures" / "qualitative_analysis" / "comparisons"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Comparison images configuration
COMPARISON_IMAGES = [
    {
        "name": "IML Lifecycle Detection Comparison",
        "dataset": "experiment_iml_lifecycle",
        "image_name": "PA171772.png",
        "gt_folder": "gt_detection",
        "pred_folder": "pred_detection_yolo11",
        "output_name": "iml_lifecycle_gt_vs_pred.png",
        "description": "Perfect 4/4 detection - GT and Pred identical"
    },
    {
        "name": "MP-IDB Species Detection Comparison",
        "dataset": "experiment_mp_idb_species",
        "image_name": "1409171742-0010-R.png",
        "gt_folder": "gt_detection",
        "pred_folder": "pred_detection_yolo11",
        "output_name": "mp_idb_species_gt_vs_pred.png",
        "description": "Perfect 5/5 detection - High density"
    },
    {
        "name": "MP-IDB Stages Detection Comparison (HERO - Gametocyte)",
        "dataset": "experiment_mp_idb_stages",
        "image_name": "1703121298-0010-G.png",
        "gt_folder": "gt_detection",
        "pred_folder": "pred_detection_yolo11",
        "output_name": "mp_idb_stages_gt_vs_pred.png",
        "description": "Perfect 2/2 gametocytes - Ultra-rare class (0.986 conf)"
    },
    {
        "name": "MD_2019 Stages Detection Comparison (HONEST - Shows FN)",
        "dataset": "experiment_md_2019_stages",
        "image_name": "Trip 064 Day 2 25-11-05 Image 5_11.png",
        "gt_folder": "gt_detection",
        "pred_folder": "pred_detection_yolo11",
        "output_name": "md_2019_stages_gt_vs_pred.png",
        "description": "7/8 detection (1 FN) - Realistic challenge"
    },
]


def add_text_label(image, text, position='top', bg_color=(255, 255, 255),
                   text_color=(0, 0, 0), font_scale=1.2, thickness=2):
    """
    Add text label overlay to image

    Args:
        image: Input image (numpy array)
        text: Text to display
        position: 'top' or 'bottom'
        bg_color: Background color (B, G, R)
        text_color: Text color (B, G, R)
        font_scale: Font size scale
        thickness: Text thickness

    Returns:
        Image with text overlay
    """
    h, w = image.shape[:2]

    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate position
    padding = 20
    label_h = text_h + baseline + 2 * padding

    if position == 'top':
        y_start = 0
        y_end = label_h
        text_y = padding + text_h
    else:  # bottom
        y_start = h - label_h
        y_end = h
        text_y = y_start + padding + text_h

    # Draw background rectangle
    cv2.rectangle(image, (0, y_start), (w, y_end), bg_color, -1)

    # Draw text (centered horizontally)
    text_x = (w - text_w) // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale,
                text_color, thickness, cv2.LINE_AA)

    return image


def create_side_by_side_comparison(gt_path, pred_path, output_path,
                                   gt_label="Ground Truth", pred_label="Prediction"):
    """
    Create side-by-side comparison image

    Args:
        gt_path: Path to ground truth image
        pred_path: Path to prediction image
        output_path: Path to save comparison image
        gt_label: Label for GT image
        pred_label: Label for Pred image

    Returns:
        True if successful, False otherwise
    """
    # Check if both images exist
    if not gt_path.exists():
        print(f"   ❌ ERROR: GT image not found: {gt_path}")
        return False

    if not pred_path.exists():
        print(f"   ❌ ERROR: Pred image not found: {pred_path}")
        return False

    # Load images
    gt_img = cv2.imread(str(gt_path))
    pred_img = cv2.imread(str(pred_path))

    if gt_img is None or pred_img is None:
        print(f"   ❌ ERROR: Failed to load images")
        return False

    # Check if images have same size
    if gt_img.shape != pred_img.shape:
        print(f"   ⚠️ WARNING: Image sizes don't match. Resizing...")
        # Resize pred to match gt
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    # Add labels to images
    gt_img_labeled = add_text_label(gt_img.copy(), gt_label, position='top',
                                    bg_color=(240, 240, 240), text_color=(0, 0, 0))
    pred_img_labeled = add_text_label(pred_img.copy(), pred_label, position='top',
                                      bg_color=(240, 240, 240), text_color=(0, 0, 0))

    # Create side-by-side comparison
    # Add small gap between images
    gap = 10
    h, w = gt_img_labeled.shape[:2]

    # Create white gap
    gap_img = np.ones((h, gap, 3), dtype=np.uint8) * 255

    # Concatenate: GT | gap | Pred
    comparison = np.hstack([gt_img_labeled, gap_img, pred_img_labeled])

    # Save comparison image
    cv2.imwrite(str(output_path), comparison)

    # Verify save
    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"   ✅ SUCCESS: Saved comparison ({file_size:,} bytes)")
        print(f"      Dimensions: {comparison.shape[1]}x{comparison.shape[0]} (WxH)")
        return True
    else:
        print(f"   ❌ ERROR: Failed to save comparison image")
        return False


def generate_all_comparisons():
    """Generate all GT vs Pred comparison images"""

    print("\n" + "="*80)
    print("[COMPARISON] GENERATING GT VS PRED COMPARISON IMAGES")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total comparisons to generate: {len(COMPARISON_IMAGES)}")

    success_count = 0
    failed_count = 0

    print("\n[GENERATE] Processing images...\n")

    for idx, img_config in enumerate(COMPARISON_IMAGES, 1):
        print(f"[{idx}/{len(COMPARISON_IMAGES)}] {img_config['name']}")
        print(f"   Description: {img_config['description']}")

        # Construct paths
        dataset_dir = RESULTS_DIR / img_config['dataset'] / "visualizations"
        gt_path = dataset_dir / img_config['gt_folder'] / img_config['image_name']
        pred_path = dataset_dir / img_config['pred_folder'] / img_config['image_name']
        output_path = OUTPUT_DIR / img_config['output_name']

        print(f"   GT: {img_config['gt_folder']}/{img_config['image_name']}")
        print(f"   Pred: {img_config['pred_folder']}/{img_config['image_name']}")

        # Create comparison
        if create_side_by_side_comparison(gt_path, pred_path, output_path):
            success_count += 1
        else:
            failed_count += 1

        print()

    # Summary
    print("="*80)
    print("[SUMMARY] COMPARISON GENERATION COMPLETE")
    print("="*80)
    print(f"   ✅ Successful: {success_count}/{len(COMPARISON_IMAGES)}")
    print(f"   ❌ Failed: {failed_count}/{len(COMPARISON_IMAGES)}")
    print(f"   📁 Output folder: {OUTPUT_DIR}")

    if success_count == len(COMPARISON_IMAGES):
        print("\n✨ ALL COMPARISON IMAGES GENERATED SUCCESSFULLY! ✨")
        print("\nNext steps:")
        print("   1. Verify comparison images visually")
        print("   2. Update IMAGE_SELECTION_REPORT.md for 3-column layout")
        print("   3. Create script to copy selected images (single + comparison)")
        print("   4. Update paper draft with 3-column figure layout")
        return True
    else:
        print("\n⚠️ SOME COMPARISONS FAILED TO GENERATE!")
        print("   Check error messages above")
        return False


def main():
    """Main execution"""

    print("\n" + "="*80)
    print("GT VS PRED COMPARISON IMAGE GENERATOR")
    print("="*80)
    print("Purpose: Generate side-by-side comparison images for 3-column paper layout")
    print("Strategy: Show GT (left) vs Pred (right) to visualize FP/FN")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    success = generate_all_comparisons()

    if success:
        print("\n" + "="*80)
        print("✨ PROCESS COMPLETE - READY FOR 3-COLUMN LAYOUT! ✨")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️ PROCESS COMPLETED WITH ERRORS")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit(main())
