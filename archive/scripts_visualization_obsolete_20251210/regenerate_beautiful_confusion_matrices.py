#!/usr/bin/env python3
"""
Regenerate Beautiful Confusion Matrices - Publication Quality

Regenerates individual confusion matrices with professional styling:
- Dual annotations (count + percentage)
- Professional journal styling
- High resolution (400 DPI)
- Color-blind friendly
- Larger, clearer labels

Usage:
    # Regenerate all confusion matrices in experiment
    python regenerate_beautiful_confusion_matrices.py --experiment-dir results/optA_20251207_233941/experiments

    # Regenerate specific dataset
    python regenerate_beautiful_confusion_matrices.py --experiment-dir results/optA_20251207_233941/experiments --dataset iml_lifecycle
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def plot_beautiful_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    dataset_name: str,
    accuracy: float,
    save_path: Path,
    dpi: int = 400
):
    """
    Plot beautiful confusion matrix with dual annotations (count + percentage).

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        model_name: Model architecture name
        dataset_name: Dataset name
        accuracy: Test accuracy
        save_path: Output path
        dpi: Resolution
    """
    plt.figure(figsize=(10, 8))

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both counts and percentages
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percent[i, j]
            if count > 0:
                row.append(f'{int(count)}\n({percent:.1f}%)')
            else:
                row.append('0\n(0.0%)')
        annotations.append(row)

    # Plot heatmap with professional styling
    sns.heatmap(
        cm,
        annot=annotations,
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )

    # Title with model and accuracy
    title = f'{dataset_name.replace("_", " ").title()} - {model_name.replace("_", " ").title()}\nTest Accuracy: {accuracy:.2f}%'
    plt.title(title, fontsize=16, fontweight='bold', pad=15)

    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')

    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()

    # Save with high DPI
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"   ✓ {save_path.name}")


def regenerate_confusion_matrix_for_model(
    exp_path: Path,
    cls_folder: Path,
    crops_dir: Path
) -> bool:
    """
    Regenerate confusion matrix for a single classification model.

    Args:
        exp_path: Experiment directory path
        cls_folder: Classification model folder
        crops_dir: Ground truth crops directory

    Returns:
        True if successful
    """
    try:
        # Get model info
        model_name = cls_folder.name.replace('cls_', '').replace('_focal', '')
        dataset_name = exp_path.name.replace('experiment_', '')

        # Check if results.csv exists
        results_csv = cls_folder / "results.csv"
        if not results_csv.exists():
            print(f"   [SKIP] {cls_folder.name} - no results.csv")
            return False

        # Load results to get best epoch
        df = pd.read_csv(results_csv)

        # Get class names from dataset folder structure
        test_dir = crops_dir / 'crops' / 'test'
        if not test_dir.exists():
            print(f"   [SKIP] {cls_folder.name} - no test directory at {test_dir}")
            return False

        class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

        # Get best epoch (based on validation accuracy)
        if 'val_acc' in df.columns:
            best_idx = df['val_acc'].idxmax()
        else:
            best_idx = df.index[-1]

        # Load best model checkpoint
        best_model_path = cls_folder / "best.pt"
        if not best_model_path.exists():
            print(f"   [SKIP] {cls_folder.name} - no best.pt")
            return False

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        from torchvision import models

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
            print(f"   [SKIP] {cls_folder.name} - unknown model type")
            return False

        # Load weights
        checkpoint = torch.load(best_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        # Setup test data loader
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_dataset = ImageFolder(crops_dir / 'crops' / 'test', transform=test_transform)
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

        # Calculate test accuracy from confusion matrix
        test_acc = (np.trace(cm) / np.sum(cm)) * 100

        # Plot beautiful confusion matrix
        output_path = cls_folder / "confusion_matrix.png"
        plot_beautiful_confusion_matrix(
            cm=cm,
            class_names=class_names,
            model_name=model_name,
            dataset_name=dataset_name,
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
        description='Regenerate beautiful confusion matrices for all models'
    )
    parser.add_argument('--experiment-dir', type=str, default=None,
                       help='Experiment directory (default: auto-detect latest)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset (e.g., iml_lifecycle, mp_idb_species)')
    parser.add_argument('--dpi', type=int, default=400,
                       help='Figure resolution (DPI)')

    args = parser.parse_args()

    # Auto-detect latest experiment if not specified
    if args.experiment_dir is None:
        results_dir = Path("results")
        if results_dir.exists():
            experiments = sorted([d for d in results_dir.iterdir()
                                if d.is_dir() and d.name.startswith('optA_')],
                               key=lambda x: x.name, reverse=True)
            if experiments:
                args.experiment_dir = str(experiments[0] / "experiments")
                print(f"[AUTO-DETECT] Using latest: {args.experiment_dir}")
            else:
                print("[ERROR] No experiments found")
                return 1
        else:
            print("[ERROR] results/ directory not found")
            return 1

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"[ERROR] Experiment directory not found: {experiment_dir}")
        return 1

    # Find ground truth crops (always in project root data/ folder)
    project_root = Path.cwd()
    crops_base = project_root / "data" / "ground_truth_crops_224"

    print("="*80)
    print("REGENERATING BEAUTIFUL CONFUSION MATRICES")
    print("="*80)
    print(f"Experiment: {experiment_dir}")
    print(f"Crops base: {crops_base}")
    print()

    # Get datasets to process
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ['iml_lifecycle', 'mp_idb_species', 'mp_idb_stages', 'md_2019_stages']

    total_regenerated = 0

    for dataset_key in datasets:
        # Map dataset key to folder names
        if dataset_key == 'iml_lifecycle':
            exp_name = 'experiment_iml_lifecycle'
            crops_name = 'lifecycle'
        elif dataset_key == 'mp_idb_species':
            exp_name = 'experiment_mp_idb_species'
            crops_name = 'species'
        elif dataset_key == 'mp_idb_stages':
            exp_name = 'experiment_mp_idb_stages'
            crops_name = 'stages'
        elif dataset_key == 'md_2019_stages':
            exp_name = 'experiment_md_2019_stages'
            crops_name = 'md_2019_stages'
        else:
            continue

        exp_path = experiment_dir / exp_name
        crops_dir = crops_base / crops_name

        if not exp_path.exists():
            print(f"[SKIP] {exp_name} - not found")
            continue

        if not crops_dir.exists():
            print(f"[SKIP] {crops_name} - crops not found")
            continue

        print(f"\n[{dataset_key.upper()}]")

        # Find all classification models
        cls_folders = sorted(exp_path.glob("cls_*_focal"))

        for cls_folder in cls_folders:
            success = regenerate_confusion_matrix_for_model(exp_path, cls_folder, crops_dir)
            if success:
                total_regenerated += 1

    print("\n" + "="*80)
    print("REGENERATION COMPLETE")
    print("="*80)
    print(f"✓ {total_regenerated} confusion matrices regenerated")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
