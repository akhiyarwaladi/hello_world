#!/usr/bin/env python3
"""
Training Curves Generator - Modular & Configurable

Generates professional publication-quality training curves (accuracy & loss).
Refactored from generate_professional_training_curves_final.py to be:
- Configurable (no hardcoded paths)
- Modular (easy to extend)
- Reusable (can be called from orchestrator)

Features:
- Color-blind friendly colors
- Journal-quality styling (400 DPI)
- Best vs Worst model comparison
- Configurable output paths
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class TrainingCurvesGenerator:
    """
    Generate professional training curves for publication.

    Architecture:
    - Configurable paths (no hardcoded values)
    - Modular design (easy to add new plot types)
    - Professional styling (journal-ready)

    Example:
        generator = TrainingCurvesGenerator(
            experiment_dir='results/optA_20251207_233941/experiments',
            output_dir='luaran/auto_generated/figures/training_curves'
        )
        generator.generate_all()
    """

    # Professional color scheme (color-blind friendly)
    COLORS = {
        'best_train': '#0173B2',      # Blue
        'best_val': '#DE8F05',        # Orange
        'worst_train': '#029E73',     # Teal/Green
        'worst_val': '#CC78BC',       # Purple
    }

    # Professional journal style
    PLOT_STYLE = {
        'font.family': 'Arial',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 400,
        'savefig.dpi': 400
    }

    def __init__(
        self,
        experiment_dir: Path,
        output_dir: Path,
        dataset_configs: Optional[Dict] = None,
        dpi: int = 400
    ):
        """
        Initialize training curves generator.

        Args:
            experiment_dir: Path to experiments directory
            output_dir: Path for output figures
            dataset_configs: Custom dataset configurations (optional)
            dpi: Figure resolution (default: 400)
        """
        self.experiment_dir = Path(experiment_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Apply professional styling
        plt.rcParams.update(self.PLOT_STYLE)
        plt.rcParams.update({'figure.dpi': dpi, 'savefig.dpi': dpi})

        # Default dataset configurations (can be overridden)
        self.dataset_configs = dataset_configs or self._get_default_configs()

    def _get_default_configs(self) -> Dict:
        """Get default dataset configurations."""
        return {
            'iml_lifecycle': {
                'name': 'IML Lifecycle',
                'best': ('efficientnet_b1', 'EfficientNet-B1'),
                'worst': ('resnet101', 'ResNet101')
            },
            'mp_idb_species': {
                'name': 'MP-IDB Species',
                'best': ('efficientnet_b1', 'EfficientNet-B1'),
                'worst': ('efficientnet_b0', 'EfficientNet-B0')
            },
            'mp_idb_stages': {
                'name': 'MP-IDB Stages',
                'best': ('resnet50', 'ResNet50'),
                'worst': ('densenet121', 'DenseNet121')
            },
            'md_2019_stages': {
                'name': 'MD_2019 Stages',
                'best': ('efficientnet_b0', 'EfficientNet-B0'),
                'worst': ('resnet101', 'ResNet101')
            }
        }

    def load_training_data(self, dataset_key: str, model_name: str) -> Optional[pd.DataFrame]:
        """
        Load training history CSV.

        Args:
            dataset_key: Dataset identifier (e.g., 'iml_lifecycle')
            model_name: Model name (e.g., 'efficientnet_b1')

        Returns:
            DataFrame with training history or None if not found
        """
        csv_path = self.experiment_dir / f"experiment_{dataset_key}" / f"cls_{model_name}_focal" / "results.csv"

        if not csv_path.exists():
            print(f"   [WARNING] {csv_path} not found!")
            return None

        return pd.read_csv(csv_path)

    def create_accuracy_figure(
        self,
        dataset_key: str,
        config: Dict,
        figsize: Tuple[float, float] = (8, 8)
    ) -> bool:
        """
        Create accuracy comparison figure (SQUARE for consistency).

        Args:
            dataset_key: Dataset identifier
            config: Dataset configuration dict
            figsize: Figure size in inches (SQUARE)

        Returns:
            True if successful, False otherwise
        """
        print(f"   [PLOT] Creating accuracy figure for {config['name']}...")

        # Load data
        best_data = self.load_training_data(dataset_key, config['best'][0])
        worst_data = self.load_training_data(dataset_key, config['worst'][0])

        if best_data is None or worst_data is None:
            print(f"   [SKIP] Missing data for {dataset_key}")
            return False

        # Create SQUARE figure for perfect consistency
        fig, ax = plt.subplots(figsize=figsize)

        # Plot best model
        ax.plot(best_data['epoch'], best_data['train_acc'],
                color=self.COLORS['best_train'], linewidth=2.5, linestyle='-',
                label=f'{config["best"][1]} (Train)', alpha=0.9)
        ax.plot(best_data['epoch'], best_data['val_acc'],
                color=self.COLORS['best_val'], linewidth=2.5, linestyle='--',
                label=f'{config["best"][1]} (Val)', alpha=0.9)

        # Plot worst model
        ax.plot(worst_data['epoch'], worst_data['train_acc'],
                color=self.COLORS['worst_train'], linewidth=2.5, linestyle='-',
                label=f'{config["worst"][1]} (Train)', alpha=0.9)
        ax.plot(worst_data['epoch'], worst_data['val_acc'],
                color=self.COLORS['worst_val'], linewidth=2.5, linestyle='--',
                label=f'{config["worst"][1]} (Val)', alpha=0.9)

        # Styling
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_ylim([20, 100])
        ax.set_xlim([0, max(best_data['epoch'].max(), worst_data['epoch'].max())])
        ax.margins(0)
        ax.grid(True, alpha=0.25, linewidth=0.8, linestyle=':')
        ax.legend(loc='lower right', frameon=True, framealpha=0.95,
                  edgecolor='black', fontsize=10, ncol=1)
        ax.tick_params(axis='both', which='major', labelsize=11, length=6)

        # Fixed padding with extra space for text labels (PERFECTLY CONSISTENT sizing)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)

        # Save with consistent sizing (NO bbox_inches='tight', NO tight_layout)
        output_path = self.output_dir / f"accuracy_{dataset_key}.png"
        plt.savefig(output_path, dpi=self.dpi, facecolor='white')
        print(f"   [SAVED] {output_path}")
        plt.close()
        return True

    def create_loss_figure(
        self,
        dataset_key: str,
        config: Dict,
        figsize: Tuple[float, float] = (8, 8)
    ) -> bool:
        """
        Create loss comparison figure (SQUARE for consistency).

        Args:
            dataset_key: Dataset identifier
            config: Dataset configuration dict
            figsize: Figure size in inches (SQUARE)

        Returns:
            True if successful, False otherwise
        """
        print(f"   [PLOT] Creating loss figure for {config['name']}...")

        # Load data
        best_data = self.load_training_data(dataset_key, config['best'][0])
        worst_data = self.load_training_data(dataset_key, config['worst'][0])

        if best_data is None or worst_data is None:
            print(f"   [SKIP] Missing data for {dataset_key}")
            return False

        # Create SQUARE figure for perfect consistency
        fig, ax = plt.subplots(figsize=figsize)

        # Plot best model
        ax.plot(best_data['epoch'], best_data['train_loss'],
                color=self.COLORS['best_train'], linewidth=2.5, linestyle='-',
                label=f'{config["best"][1]} (Train)', alpha=0.9)
        ax.plot(best_data['epoch'], best_data['val_loss'],
                color=self.COLORS['best_val'], linewidth=2.5, linestyle='--',
                label=f'{config["best"][1]} (Val)', alpha=0.9)

        # Plot worst model
        ax.plot(worst_data['epoch'], worst_data['train_loss'],
                color=self.COLORS['worst_train'], linewidth=2.5, linestyle='-',
                label=f'{config["worst"][1]} (Train)', alpha=0.9)
        ax.plot(worst_data['epoch'], worst_data['val_loss'],
                color=self.COLORS['worst_val'], linewidth=2.5, linestyle='--',
                label=f'{config["worst"][1]} (Val)', alpha=0.9)

        # Styling
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=13, fontweight='bold')

        # Set reasonable y-axis limits
        max_loss = max(best_data['val_loss'].max(), worst_data['val_loss'].max())
        ax.set_ylim([0, min(max_loss * 1.1, 2.0)])

        ax.grid(True, alpha=0.25, linewidth=0.8, linestyle=':')
        ax.legend(loc='upper right', frameon=True, framealpha=0.95,
                  edgecolor='black', fontsize=10, ncol=1)
        ax.tick_params(axis='both', which='major', labelsize=11, length=6)

        # Fixed padding with extra space for text labels (PERFECTLY CONSISTENT sizing)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)

        # Save with consistent sizing (NO bbox_inches='tight', NO tight_layout)
        output_path = self.output_dir / f"loss_{dataset_key}.png"
        plt.savefig(output_path, dpi=self.dpi, facecolor='white')
        print(f"   [SAVED] {output_path}")
        plt.close()
        return True

    def generate_all(self, plot_types: List[str] = ['accuracy']) -> Dict[str, int]:
        """
        Generate all training curves.

        Args:
            plot_types: List of plot types to generate ['accuracy', 'loss']

        Returns:
            Dict with success counts per plot type
        """
        print("\n" + "="*80)
        print("GENERATING PROFESSIONAL TRAINING CURVES")
        print("="*80)
        print(f"Output: {self.output_dir}")
        print(f"Plot types: {', '.join(plot_types)}")
        print(f"Datasets: {len(self.dataset_configs)}")
        print()

        results = {plot_type: 0 for plot_type in plot_types}

        for dataset_key, config in self.dataset_configs.items():
            print(f"\n[{config['name']}]")
            print(f"  Best:  {config['best'][1]}")
            print(f"  Worst: {config['worst'][1]}")

            if 'accuracy' in plot_types:
                if self.create_accuracy_figure(dataset_key, config):
                    results['accuracy'] += 1

            if 'loss' in plot_types:
                if self.create_loss_figure(dataset_key, config):
                    results['loss'] += 1

        print("\n" + "="*80)
        print("GENERATION COMPLETE")
        print("="*80)
        for plot_type, count in results.items():
            print(f"  {plot_type.capitalize()}: {count}/{len(self.dataset_configs)} figures")
        print(f"  Location: {self.output_dir}")
        print("="*80)

        return results


def main():
    """Standalone mode with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate professional training curves for publication'
    )
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Path to experiments directory')
    parser.add_argument('--output-dir', type=str, default='luaran/auto_generated/figures/training_curves',
                       help='Output directory for figures')
    parser.add_argument('--plot-types', nargs='+', default=['accuracy'],
                       choices=['accuracy', 'loss'],
                       help='Types of plots to generate')
    parser.add_argument('--dpi', type=int, default=400,
                       help='Figure resolution (DPI)')

    args = parser.parse_args()

    # Create generator
    generator = TrainingCurvesGenerator(
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
        dpi=args.dpi
    )

    # Generate all figures
    generator.generate_all(plot_types=args.plot_types)


if __name__ == '__main__':
    main()
