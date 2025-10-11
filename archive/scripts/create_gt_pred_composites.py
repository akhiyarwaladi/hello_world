"""
Create GT vs Prediction side-by-side composite images for JICEST paper
Generates 4 figures:
1. MP-IDB Stages - Detection GT vs Pred
2. MP-IDB Stages - Classification GT vs Pred
3. MP-IDB Species - Detection GT vs Pred
4. MP-IDB Species - Classification GT vs Pred
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def create_side_by_side_composite(gt_path, pred_path, output_path, title_gt, title_pred, main_title):
    """
    Create side-by-side comparison image with titles and border
    """
    # Read images
    img_gt = cv2.imread(str(gt_path))
    img_pred = cv2.imread(str(pred_path))

    if img_gt is None or img_pred is None:
        print(f"Error reading images: {gt_path} or {pred_path}")
        return False

    # Convert BGR to RGB for matplotlib
    img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
    img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.98)

    # Left: Ground Truth
    axes[0].imshow(img_gt_rgb)
    axes[0].set_title(title_gt, fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Add border
    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('blue')
        spine.set_linewidth(3)

    # Right: Prediction
    axes[1].imshow(img_pred_rgb)
    axes[1].set_title(title_pred, fontsize=14, fontweight='bold', pad=10)
    axes[1].axis('off')

    # Add border
    for spine in axes[1].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('green')
        spine.set_linewidth(3)

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save high-resolution
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] Created: {output_path}")
    return True

def main():
    """Generate all 4 composite images"""

    # Base paths
    base_results = Path("results/optA_20251007_134458/experiments")
    output_dir = Path("luaran/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Selected best examples (you can customize these)
    selected_images = {
        'stages': {
            'detection': '1704282807-0021-T_G_R.png',      # Complex: 17 parasites with multiple stages
            'classification': '1704282807-0021-T_G_R.png'   # Same image, shows classification results
        },
        'species': {
            'detection': '1704282807-0021-T_G_R.png',      # Complex: 17 P. falciparum parasites
            'classification': '1709041080-0036-S_R.png'     # Mixed species: Schizont + P. vivax
        }
    }

    configs = [
        # 1. MP-IDB Stages - Detection
        {
            'dataset': 'mp_idb_stages',
            'type': 'detection',
            'image': selected_images['stages']['detection'],
            'title_gt': 'Ground Truth Detection (Expert Annotations)',
            'title_pred': 'YOLOv11 Detection (Automated)',
            'main_title': 'MP-IDB Stages: Parasite Detection Performance (YOLOv11)',
            'output': 'figure9a_stages_detection_gt_vs_pred.png'
        },
        # 2. MP-IDB Stages - Classification
        {
            'dataset': 'mp_idb_stages',
            'type': 'classification',
            'image': selected_images['stages']['classification'],
            'title_gt': 'Ground Truth Classification (Expert Labels)',
            'title_pred': 'EfficientNet-B1 Classification (Automated)',
            'main_title': 'MP-IDB Stages: Lifecycle Stage Classification (EfficientNet-B1)',
            'output': 'figure9b_stages_classification_gt_vs_pred.png'
        },
        # 3. MP-IDB Species - Detection
        {
            'dataset': 'mp_idb_species',
            'type': 'detection',
            'image': selected_images['species']['detection'],
            'title_gt': 'Ground Truth Detection (Expert Annotations)',
            'title_pred': 'YOLOv11 Detection (Automated)',
            'main_title': 'MP-IDB Species: Parasite Detection Performance (YOLOv11)',
            'output': 'figure9c_species_detection_gt_vs_pred.png'
        },
        # 4. MP-IDB Species - Classification
        {
            'dataset': 'mp_idb_species',
            'type': 'classification',
            'image': selected_images['species']['classification'],
            'title_gt': 'Ground Truth Classification (Expert Labels)',
            'title_pred': 'EfficientNet-B1 Classification (Automated)',
            'main_title': 'MP-IDB Species: Plasmodium Species Classification (EfficientNet-B1)',
            'output': 'figure9d_species_classification_gt_vs_pred.png'
        }
    ]

    print("\n" + "="*80)
    print("CREATING GT vs PRED COMPOSITE IMAGES FOR JICEST PAPER")
    print("="*80 + "\n")

    success_count = 0
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/4] Processing: {config['main_title']}")
        print(f"  Dataset: {config['dataset']}")
        print(f"  Type: {config['type']}")
        print(f"  Image: {config['image']}")

        # Build paths
        experiment_path = base_results / f"experiment_{config['dataset']}"
        det_cls_path = experiment_path / "detection_classification_figures" / "det_yolo11_cls_efficientnet_b1_focal"

        gt_path = det_cls_path / f"gt_{config['type']}" / config['image']
        pred_path = det_cls_path / f"pred_{config['type']}" / config['image']
        output_path = output_dir / config['output']

        # Check if files exist
        if not gt_path.exists():
            print(f"  [ERROR] GT image not found: {gt_path}")
            continue
        if not pred_path.exists():
            print(f"  [ERROR] Pred image not found: {pred_path}")
            continue

        # Create composite
        if create_side_by_side_composite(
            gt_path, pred_path, output_path,
            config['title_gt'], config['title_pred'], config['main_title']
        ):
            success_count += 1
            print(f"  [SUCCESS] Saved to: {output_path}")

    print("\n" + "="*80)
    print(f"COMPLETED: {success_count}/4 composite images created successfully")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
