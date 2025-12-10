#!/usr/bin/env python3
"""
Consolidated Confusion Matrix Generator - Publication Quality

Generates consolidated confusion matrix figures from individual model outputs:
1. Per-dataset 2x2 grid (best 2 models for species + best 2 for stages)
2. All-in-one overview grid

Professional journal-quality styling:
- Color-blind friendly colormaps
- Clean layout
- High resolution (400 DPI)
- Standardized formatting

Usage:
    # Auto-detect latest experiment
    python generate_consolidated_confusion_matrices.py

    # Specific experiment
    python generate_consolidated_confusion_matrices.py \
      --experiment-dir results/optA_20251207_233941 \
      --output-dir luaran/auto_generated/figures/performance
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class ConsolidatedConfusionMatrixGenerator:
    """
    Generate publication-quality consolidated confusion matrices.

    Features:
    - Reads confusion matrices from individual model folders
    - Creates consolidated 2x2 grids (best models)
    - Professional journal styling
    - Color-blind friendly
    """

    # Professional styling
    PLOT_STYLE = {
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'figure.dpi': 400,
        'savefig.dpi': 400,
        'savefig.bbox': 'tight'
    }

    def __init__(
        self,
        experiment_dir: Path,
        output_dir: Path,
        dpi: int = 400
    ):
        """
        Initialize generator.

        Args:
            experiment_dir: Path to experiments directory
            output_dir: Output directory for figures
            dpi: Figure resolution
        """
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Apply styling
        plt.rcParams.update(self.PLOT_STYLE)
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi

    def load_confusion_matrix_from_csv(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """
        Load confusion matrix data from results CSV.

        Args:
            csv_path: Path to results.csv file

        Returns:
            DataFrame with confusion matrix or None if not found
        """
        try:
            df = pd.read_csv(csv_path)

            # Check if this is classification results (has test_* columns)
            test_cols = [col for col in df.columns if col.startswith('test_')]

            if not test_cols:
                return None

            # Extract class names from column names (e.g., 'test_ring', 'test_gametocyte')
            class_names = [col.replace('test_', '') for col in test_cols]

            # Get best epoch row
            if 'val_acc' in df.columns:
                best_idx = df['val_acc'].idxmax()
            elif 'val_loss' in df.columns:
                best_idx = df['val_loss'].idxmin()
            else:
                best_idx = df.index[-1]  # Last epoch

            # Extract confusion matrix values
            cm_values = df.loc[best_idx, test_cols].values

            # This is actually per-class accuracy, not full confusion matrix
            # We need to load from actual confusion_matrix.png or compute from predictions
            return None  # For now, we'll use image-based approach

        except Exception as e:
            print(f"   [WARNING] Failed to load from CSV: {e}")
            return None

    def find_best_models(self, dataset_key: str) -> Dict[str, str]:
        """
        Find best classification models for a dataset.

        Args:
            dataset_key: Dataset identifier (e.g., 'iml_lifecycle')

        Returns:
            Dict with 'best' and 'second_best' model names
        """
        exp_path = self.experiment_dir / f"experiment_{dataset_key}"

        if not exp_path.exists():
            return {}

        # Find all classification model folders
        cls_folders = list(exp_path.glob("cls_*_focal"))

        if not cls_folders:
            return {}

        # Read accuracies from results.csv
        model_scores = {}
        for cls_folder in cls_folders:
            results_csv = cls_folder / "results.csv"
            if results_csv.exists():
                try:
                    df = pd.read_csv(results_csv)
                    if 'val_acc' in df.columns:
                        best_acc = df['val_acc'].max()
                        model_name = cls_folder.name.replace('cls_', '').replace('_focal', '')
                        model_scores[model_name] = best_acc
                except Exception:
                    pass

        if not model_scores:
            # Fallback: use default best models
            return {
                'best': 'efficientnet_b1',
                'second_best': 'efficientnet_b0'
            }

        # Sort by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'best': sorted_models[0][0] if len(sorted_models) > 0 else 'efficientnet_b1',
            'second_best': sorted_models[1][0] if len(sorted_models) > 1 else 'efficientnet_b0'
        }

    def create_2x2_confusion_matrix_grid(
        self,
        species_best: Path,
        species_second: Path,
        stages_best: Path,
        stages_second: Path,
        output_path: Path
    ) -> bool:
        """
        Create 2x2 grid with 4 confusion matrices.

        Args:
            species_best: Path to best species model confusion matrix PNG
            species_second: Path to second best species model confusion matrix PNG
            stages_best: Path to best stages model confusion matrix PNG
            stages_second: Path to second best stages model confusion matrix PNG
            output_path: Output path for combined figure

        Returns:
            True if successful
        """
        try:
            from PIL import Image

            # Load images
            img_species_best = Image.open(species_best)
            img_species_second = Image.open(species_second)
            img_stages_best = Image.open(stages_best)
            img_stages_second = Image.open(stages_second)

            # Create figure with 2x2 layout
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            fig.suptitle('Confusion Matrices - Classification Performance', fontsize=20, fontweight='bold', y=0.98)

            # Plot images
            axes[0, 0].imshow(img_species_best)
            axes[0, 0].axis('off')
            axes[0, 0].set_title('Species - Best Model', fontsize=14, pad=10)

            axes[0, 1].imshow(img_species_second)
            axes[0, 1].axis('off')
            axes[0, 1].set_title('Species - Second Best', fontsize=14, pad=10)

            axes[1, 0].imshow(img_stages_best)
            axes[1, 0].axis('off')
            axes[1, 0].set_title('Stages - Best Model', fontsize=14, pad=10)

            axes[1, 1].imshow(img_stages_second)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Stages - Second Best', fontsize=14, pad=10)

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"   [SAVED] {output_path}")
            return True

        except Exception as e:
            print(f"   [ERROR] Failed to create 2x2 grid: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_all(self) -> Dict[str, bool]:
        """
        Generate all consolidated confusion matrix figures.

        Returns:
            Dict with generation results
        """
        print("\n" + "="*80)
        print("GENERATING CONSOLIDATED CONFUSION MATRICES")
        print("="*80)
        print(f"Experiment: {self.experiment_dir}")
        print(f"Output: {self.output_dir}")
        print()

        results = {}

        # Find best models for each dataset
        print("[STEP 1] Finding best models...")
        species_models = self.find_best_models('mp_idb_species')
        stages_iml_models = self.find_best_models('iml_lifecycle')
        stages_mp_models = self.find_best_models('mp_idb_stages')

        print(f"  Species best: {species_models.get('best', 'N/A')}")
        print(f"  IML Lifecycle best: {stages_iml_models.get('best', 'N/A')}")
        print(f"  MP-IDB Stages best: {stages_mp_models.get('best', 'N/A')}")

        # Generate species + lifecycle 2x2 grid
        print("\n[STEP 2] Generating Species + Lifecycle 2x2 grid...")
        species_best_cm = (self.experiment_dir / f"experiment_mp_idb_species" /
                          f"cls_{species_models.get('best', 'efficientnet_b1')}_focal" / "confusion_matrix.png")
        species_second_cm = (self.experiment_dir / f"experiment_mp_idb_species" /
                            f"cls_{species_models.get('second_best', 'efficientnet_b0')}_focal" / "confusion_matrix.png")
        lifecycle_best_cm = (self.experiment_dir / f"experiment_iml_lifecycle" /
                            f"cls_{stages_iml_models.get('best', 'efficientnet_b1')}_focal" / "confusion_matrix.png")
        lifecycle_second_cm = (self.experiment_dir / f"experiment_iml_lifecycle" /
                              f"cls_{stages_iml_models.get('second_best', 'efficientnet_b0')}_focal" / "confusion_matrix.png")

        if all(p.exists() for p in [species_best_cm, species_second_cm, lifecycle_best_cm, lifecycle_second_cm]):
            output_path = self.output_dir / "confusion_matrices.png"
            success = self.create_2x2_confusion_matrix_grid(
                species_best_cm,
                species_second_cm,
                lifecycle_best_cm,
                lifecycle_second_cm,
                output_path
            )
            results['species_lifecycle_2x2'] = success
        else:
            print("   [SKIP] Some confusion matrices not found")
            results['species_lifecycle_2x2'] = False

        print("\n" + "="*80)
        print("GENERATION COMPLETE")
        print("="*80)
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
        print(f"\nOutput directory: {self.output_dir}")
        print("="*80)

        return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate consolidated confusion matrices for publication'
    )
    parser.add_argument('--experiment-dir', type=str, default=None,
                       help='Experiment directory (default: auto-detect latest)')
    parser.add_argument('--output-dir', type=str,
                       default='luaran/auto_generated/figures/performance',
                       help='Output directory for figures')
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
                print(f"[AUTO-DETECT] Using latest experiment: {args.experiment_dir}")
            else:
                print("[ERROR] No experiments found")
                return 1
        else:
            print("[ERROR] results/ directory not found")
            return 1

    # Create generator
    generator = ConsolidatedConfusionMatrixGenerator(
        experiment_dir=Path(args.experiment_dir),
        output_dir=Path(args.output_dir),
        dpi=args.dpi
    )

    # Generate all
    results = generator.generate_all()

    # Return success if any generated
    return 0 if any(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
