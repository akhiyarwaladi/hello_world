#!/usr/bin/env python3
"""
Detection Training Curves Generator - YOLO Models
Generates professional publication-quality training curves for object detection.

Features:
- mAP curves (mAP50, mAP50-95)
- Loss curves (box + cls + dfl combined)
- Best vs Worst YOLO comparison
- Journal-quality styling (400 DPI)
- Color-blind friendly colors
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class DetectionTrainingCurvesGenerator:
    """
    Generate professional detection training curves for YOLO models.

    Handles YOLO-specific metrics:
    - mAP50 (primary metric)
    - mAP50-95 (COCO metric)
    - Combined loss (box_loss + cls_loss + dfl_loss)
    """

    # Professional color scheme (color-blind friendly)
    COLORS = {
        'yolo11_train': '#0173B2',      # Blue
        'yolo11_val': '#DE8F05',        # Orange
        'yolo10_train': '#029E73',     # Teal/Green
        'yolo10_val': '#CC78BC',       # Purple
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
        Initialize detection training curves generator.

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

        # Default dataset configurations
        self.dataset_configs = dataset_configs or self._get_default_configs()

    def _get_default_configs(self) -> Dict:
        """Get default dataset configurations for detection."""
        return {
            'iml_lifecycle': {
                'name': 'IML Lifecycle',
                'best': ('yolo11', 'YOLOv11'),
                'worst': ('yolo10', 'YOLOv10')
            },
            'mp_idb_species': {
                'name': 'MP-IDB Species',
                'best': ('yolo11', 'YOLOv11'),
                'worst': ('yolo10', 'YOLOv10')
            },
            'mp_idb_stages': {
                'name': 'MP-IDB Stages',
                'best': ('yolo11', 'YOLOv11'),
                'worst': ('yolo10', 'YOLOv10')
            },
            'md_2019_stages': {
                'name': 'MD_2019 Stages',
                'best': ('yolo11', 'YOLOv11'),
                'worst': ('yolo10', 'YOLOv10')
            }
        }

    def load_detection_data(self, dataset_key: str, model_name: str) -> Optional[pd.DataFrame]:
        """
        Load YOLO detection training history CSV.

        Args:
            dataset_key: Dataset identifier (e.g., 'iml_lifecycle')
            model_name: Model name (e.g., 'yolo11')

        Returns:
            DataFrame with training history or None if not found
        """
        csv_path = self.experiment_dir / f"experiment_{dataset_key}" / f"det_{model_name}" / "results.csv"

        if not csv_path.exists():
            print(f"   [WARNING] {csv_path} not found!")
            return None

        df = pd.read_csv(csv_path)

        # Calculate combined metrics for easier plotting
        # Note: YOLO doesn't have separate train mAP, only validation
        # So we'll use validation metrics and losses for both
        if 'train/box_loss' in df.columns:
            df['train_loss_combined'] = (
                df['train/box_loss'] +
                df['train/cls_loss'] +
                df['train/dfl_loss']
            )

        if 'val/box_loss' in df.columns:
            df['val_loss_combined'] = (
                df['val/box_loss'] +
                df['val/cls_loss'] +
                df['val/dfl_loss']
            )

        return df

    def create_map_figure(
        self,
        dataset_key: str,
        config: Dict,
        figsize: Tuple[float, float] = (4, 4),
        metric: str = 'mAP50'
    ) -> bool:
        """
        Create mAP comparison figure (best vs worst).

        Args:
            dataset_key: Dataset identifier
            config: Dataset configuration dict
            figsize: Figure size in inches
            metric: Which mAP to use ('mAP50' or 'mAP50-95')

        Returns:
            True if successful, False otherwise
        """
        metric_col = 'metrics/mAP50(B)' if metric == 'mAP50' else 'metrics/mAP50-95(B)'
        print(f"   [PLOT] Creating {metric} figure for {config['name']}...")

        # Load data
        best_data = self.load_detection_data(dataset_key, config['best'][0])
        worst_data = self.load_detection_data(dataset_key, config['worst'][0])

        if best_data is None or worst_data is None:
            print(f"   [SKIP] Missing data for {dataset_key}")
            return False

        if metric_col not in best_data.columns:
            print(f"   [SKIP] {metric_col} not found in data")
            return False

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # YOLO only has validation mAP, no training mAP
        # Plot validation mAP for both models
        ax.plot(best_data['epoch'], best_data[metric_col] * 100,
                color=self.COLORS['yolo11_val'], linewidth=2.5, linestyle='-',
                label=f'{config["best"][1]}', alpha=0.9, marker='o', markersize=3, markevery=5)

        ax.plot(worst_data['epoch'], worst_data[metric_col] * 100,
                color=self.COLORS['yolo10_val'], linewidth=2.5, linestyle='-',
                label=f'{config["worst"][1]}', alpha=0.9, marker='s', markersize=3, markevery=5)

        # Styling
        ax.set_ylim([0, 100])
        ax.set_xlim([0, max(best_data['epoch'].max(), worst_data['epoch'].max())])
        ax.margins(0)
        ax.grid(True, alpha=0.25, linewidth=0.8, linestyle=':')
        ax.legend(loc='lower right', frameon=True, framealpha=0.95,
                  edgecolor='black', fontsize=10, ncol=1)
        ax.tick_params(axis='both', which='major', labelsize=11, length=6)
        ax.set_ylabel(f'{metric} (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')

        # Remove padding
        plt.subplots_adjust(left=0.12, right=0.995, top=0.995, bottom=0.08)

        # Save
        output_path = self.output_dir / f"map_{dataset_key}.png"
        plt.savefig(output_path, dpi=self.dpi, facecolor='white')
        print(f"   [SAVED] {output_path}")
        plt.close()
        return True

    def create_loss_figure(
        self,
        dataset_key: str,
        config: Dict,
        figsize: Tuple[float, float] = (5, 4.3)
    ) -> bool:
        """
        Create combined loss comparison figure (best vs worst).

        Args:
            dataset_key: Dataset identifier
            config: Dataset configuration dict
            figsize: Figure size in inches

        Returns:
            True if successful, False otherwise
        """
        print(f"   [PLOT] Creating loss figure for {config['name']}...")

        # Load data
        best_data = self.load_detection_data(dataset_key, config['best'][0])
        worst_data = self.load_detection_data(dataset_key, config['worst'][0])

        if best_data is None or worst_data is None:
            print(f"   [SKIP] Missing data for {dataset_key}")
            return False

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot combined losses (train + val)
        ax.plot(best_data['epoch'], best_data['train_loss_combined'],
                color=self.COLORS['yolo11_train'], linewidth=2.5, linestyle='-',
                label=f'{config["best"][1]} (Train)', alpha=0.9)
        ax.plot(best_data['epoch'], best_data['val_loss_combined'],
                color=self.COLORS['yolo11_val'], linewidth=2.5, linestyle='--',
                label=f'{config["best"][1]} (Val)', alpha=0.9)

        ax.plot(worst_data['epoch'], worst_data['train_loss_combined'],
                color=self.COLORS['yolo10_train'], linewidth=2.5, linestyle='-',
                label=f'{config["worst"][1]} (Train)', alpha=0.9)
        ax.plot(worst_data['epoch'], worst_data['val_loss_combined'],
                color=self.COLORS['yolo10_val'], linewidth=2.5, linestyle='--',
                label=f'{config["worst"][1]} (Val)', alpha=0.9)

        # Styling
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Combined Loss', fontsize=13, fontweight='bold')

        # Set reasonable y-axis limits
        all_losses = pd.concat([
            best_data['val_loss_combined'],
            worst_data['val_loss_combined']
        ])
        max_loss = all_losses.max()
        ax.set_ylim([0, min(max_loss * 1.1, 15.0)])

        ax.grid(True, alpha=0.25, linewidth=0.8, linestyle=':')
        ax.legend(loc='upper right', frameon=True, framealpha=0.95,
                  edgecolor='black', fontsize=10, ncol=1)
        ax.tick_params(axis='both', which='major', labelsize=11, length=6)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"loss_det_{dataset_key}.png"
        plt.savefig(output_path, dpi=self.dpi, facecolor='white')
        print(f"   [SAVED] {output_path}")
        plt.close()
        return True

    def generate_all(self, plot_types: List[str] = ['map', 'loss']) -> Dict[str, int]:
        """
        Generate all detection training curves.

        Args:
            plot_types: List of plot types to generate ['map', 'loss']

        Returns:
            Dict with success counts per plot type
        """
        print("\n" + "="*80)
        print("GENERATING DETECTION TRAINING CURVES (YOLO)")
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

            if 'map' in plot_types:
                if self.create_map_figure(dataset_key, config):
                    results['map'] += 1

            if 'loss' in plot_types:
                if self.create_loss_figure(dataset_key, config):
                    results['loss'] += 1

        print("\n" + "="*80)
        print("DETECTION CURVES GENERATION COMPLETE")
        print("="*80)
        for plot_type, count in results.items():
            print(f"  {plot_type.upper()}: {count}/{len(self.dataset_configs)} figures")
        print(f"  Location: {self.output_dir}")
        print("="*80)

        return results


def main():
    """Standalone mode with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate professional detection training curves for YOLO models'
    )
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Path to experiments directory')
    parser.add_argument('--output-dir', type=str,
                       default='visualization_outputs/training_curves',
                       help='Output directory for figures')
    parser.add_argument('--plot-types', nargs='+', default=['map', 'loss'],
                       choices=['map', 'loss'],
                       help='Types of plots to generate')
    parser.add_argument('--dpi', type=int, default=400,
                       help='Figure resolution (DPI)')

    args = parser.parse_args()

    # Create generator
    generator = DetectionTrainingCurvesGenerator(
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
        dpi=args.dpi
    )

    # Generate all figures
    generator.generate_all(plot_types=args.plot_types)


if __name__ == '__main__':
    main()
