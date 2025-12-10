#!/usr/bin/env python3
"""
Regenerate Publication-Quality Confusion Matrices

Features matching previous beautiful version:
- NO 45-degree axis rotation (rotation=0 for all labels)
- Dataset-specific color schemes (different colormap per dataset)
- Dual annotations (count + percentage)
- High resolution (400 DPI)
- Professional journal styling

Color schemes:
- IML Lifecycle: Blues (blue theme)
- MP-IDB Species: Greens (green theme)
- MP-IDB Stages: Purples (purple theme)
- MD-2019 Stages: Oranges (orange theme)

Usage:
    python regenerate_publication_quality_confusion_matrices.py
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# Dataset-specific color schemes (NO 45-degree rotation!)
DATASET_COLORS = {
    'iml_lifecycle': 'Blues',
    'mp_idb_species': 'Greens',
    'mp_idb_stages': 'Purples',
    'md_2019_stages': 'Oranges'
}


def plot_publication_quality_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    dataset_name: str,
    accuracy: float,
    save_path: Path,
    dpi: int = 400
):
    """
    Plot publication-quality confusion matrix.

    KEY FEATURES:
    - NO 45-degree rotation (rotation=0)
    - Dataset-specific colormap
    - Dual annotations (count + percentage)
    - Large figure size for readability
    - Professional journal styling

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        model_name: Model architecture name
        dataset_name: Dataset name (determines color scheme)
        accuracy: Test accuracy
        save_path: Output path
        dpi: Resolution (default 400)
    """
    # Get dataset-specific colormap
    cmap = DATASET_COLORS.get(dataset_name, 'Blues')

    # Calculate percentages for dual annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure with FIXED size for consistency across all confusion matrices
    # Reduced width to 8 to minimize right white space
    fig, ax = plt.subplots(figsize=(8, 8))  # Square, compact

    # Plot heatmap with dataset-specific colors (NO annot, we'll add manually)
    sns.heatmap(
        cm,
        annot=False,  # We'll add manual annotations
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={
            'label': '',     # No label (removed "Count")
            'shrink': 0.82,  # Optimized colorbar size
            'aspect': 20,    # Ultra-thin colorbar
            'pad': 0.01      # Ultra-minimal gap
        },
        linewidths=1.0,
        linecolor='white',
        square=True,
        ax=ax
    )

    # Manual annotations with different font sizes for count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]

            # Format percentage: remove .0 for whole numbers
            percent_str = f'{percent:.1f}%'
            if percent_str.endswith('.0%'):
                percent_str = percent_str[:-3] + '%'

            # Plot count (larger, bold)
            ax.text(j + 0.5, i + 0.4, f'{int(count)}',
                   ha='center', va='center',
                   fontsize=22, fontweight='bold',
                   color='white' if cm[i, j] > cm.max() / 2 else 'black')

            # Plot percentage (smaller, regular)
            ax.text(j + 0.5, i + 0.65, f'({percent_str})',
                   ha='center', va='center',
                   fontsize=16, fontweight='normal',
                   color='white' if cm[i, j] > cm.max() / 2 else 'black')

    # Title with model and accuracy
    dataset_display = dataset_name.replace('_', ' ').title()
    model_display = model_name.replace('_', ' ').title()
    title = f'{dataset_display} - {model_display}\nTest Accuracy: {accuracy:.2f}%'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=12)

    # Labels - Y-axis rotated 90° to save space
    ax.set_xlabel('Predicted Class', fontsize=16, fontweight='bold', labelpad=6)
    ax.set_ylabel('True Class', fontsize=16, fontweight='bold', labelpad=6)

    # X-axis labels: horizontal (rotation=0), moderate font
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=14, ha='center', fontweight='bold')
    # Y-axis labels: rotated 90° for space efficiency, moderate font
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, fontsize=14, va='center', fontweight='bold')

    # FIXED margins for consistent sizing (no bbox_inches='tight')
    # Ultra-minimal: more space bottom (for xlabel), zero right
    # Right: 1.0, Top: 0.92, Bottom: 0.09 (increased for xlabel), Left: 0.10
    plt.subplots_adjust(left=0.10, bottom=0.09, right=1.0, top=0.92)

    # Save with FIXED size (no bbox_inches='tight' to ensure consistent dimensions)
    plt.savefig(save_path, dpi=dpi, facecolor='white')
    plt.close()

    print(f"   ✓ {save_path.name} ({cmap} colormap)")


def regenerate_confusion_matrix_for_model(
    exp_path: Path,
    cls_folder: Path,
    crops_dir: Path,
    dataset_key: str
) -> bool:
    """
    Regenerate confusion matrix for a single model.

    Args:
        exp_path: Experiment directory
        cls_folder: Classification model folder
        crops_dir: Ground truth crops directory
        dataset_key: Dataset identifier (for color scheme)

    Returns:
        True if successful
    """
    try:
        model_name = cls_folder.name.replace('cls_', '').replace('_focal', '')
        dataset_name = exp_path.name.replace('experiment_', '')

        # Check requirements
        results_csv = cls_folder / "results.csv"
        best_model_path = cls_folder / "best.pt"

        if not results_csv.exists() or not best_model_path.exists():
            print(f"   [SKIP] {cls_folder.name} - missing files")
            return False

        # Get class names from crops directory
        test_dir = crops_dir / 'crops' / 'test'
        if not test_dir.exists():
            print(f"   [SKIP] {cls_folder.name} - no test directory")
            return False

        class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model architecture
        if 'efficientnet' in model_name:
            if 'b0' in model_name:
                model = models.efficientnet_b0(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
            elif 'b1' in model_name:
                model = models.efficientnet_b1(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
            elif 'b2' in model_name:
                model = models.efficientnet_b2(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
        elif 'resnet50' in model_name:
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        elif 'resnet101' in model_name:
            model = models.resnet101(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        elif 'densenet121' in model_name:
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
        else:
            print(f"   [SKIP] {cls_folder.name} - unknown model")
            return False

        # Load weights
        checkpoint = torch.load(best_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        # Setup test data
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_dataset = ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Get predictions
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        test_acc = (np.trace(cm) / np.sum(cm)) * 100

        # Plot with publication quality styling
        output_path = cls_folder / "confusion_matrix.png"
        plot_publication_quality_confusion_matrix(
            cm=cm,
            class_names=class_names,
            model_name=model_name,
            dataset_name=dataset_key,
            accuracy=test_acc,
            save_path=output_path,
            dpi=400
        )

        return True

    except Exception as e:
        print(f"   [ERROR] {cls_folder.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Regenerate publication-quality confusion matrices (NO rotation, dataset colors)'
    )
    parser.add_argument('--experiment-dir', type=str, default=None,
                       help='Experiment directory')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to process')

    args = parser.parse_args()

    # Auto-detect experiment
    if args.experiment_dir is None:
        results_dir = Path("results")
        if results_dir.exists():
            experiments = sorted([d for d in results_dir.iterdir()
                                if d.is_dir() and d.name.startswith('optA_')],
                               key=lambda x: x.name, reverse=True)
            if experiments:
                args.experiment_dir = str(experiments[0] / "experiments")
            else:
                print("[ERROR] No experiments found")
                return 1

    experiment_dir = Path(args.experiment_dir)
    project_root = Path.cwd()
    crops_base = project_root / "data" / "ground_truth_crops_224"

    print("="*80)
    print("REGENERATING PUBLICATION-QUALITY CONFUSION MATRICES")
    print("="*80)
    print("Features:")
    print("  ✓ NO 45-degree axis rotation (rotation=0)")
    print("  ✓ Dataset-specific colormaps:")
    for dataset, cmap in DATASET_COLORS.items():
        print(f"    - {dataset}: {cmap}")
    print("  ✓ Dual annotations (count + percentage)")
    print("  ✓ High resolution (400 DPI)")
    print("="*80)
    print()

    # Datasets to process
    if args.dataset:
        datasets_to_process = {args.dataset: args.dataset.replace('iml_', '').replace('mp_idb_', '').replace('md_2019_', '')}
    else:
        datasets_to_process = {
            'iml_lifecycle': 'lifecycle',
            'mp_idb_species': 'species',
            'mp_idb_stages': 'stages',
            'md_2019_stages': 'md_2019_stages'
        }

    total_regenerated = 0

    for dataset_key, crops_name in datasets_to_process.items():
        exp_path = experiment_dir / f"experiment_{dataset_key}"
        # Use crops from experiment directory instead of data/ground_truth_crops_224/
        crops_dir = exp_path / "crops_gt_crops"

        if not exp_path.exists():
            print(f"[SKIP] {dataset_key} - experiment not found")
            continue

        if not crops_dir.exists():
            print(f"[SKIP] {dataset_key} - crops not found")
            continue

        print(f"\n[{dataset_key.upper()}] - {DATASET_COLORS[dataset_key]} colormap")

        cls_folders = sorted(exp_path.glob("cls_*_focal"))
        for cls_folder in cls_folders:
            success = regenerate_confusion_matrix_for_model(exp_path, cls_folder, crops_dir, dataset_key)
            if success:
                total_regenerated += 1

    print("\n" + "="*80)
    print("REGENERATION COMPLETE")
    print("="*80)
    print(f"✓ {total_regenerated} publication-quality confusion matrices regenerated")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
