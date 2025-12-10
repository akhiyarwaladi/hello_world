#!/usr/bin/env python3
"""
UNIFIED VISUALIZATION ORCHESTRATOR
==================================

Centralized, modular, extensible visualization pipeline for malaria detection project.

Features:
- Auto-detects experiment structure
- Generates ALL visualizations (detection, classification, training curves)
- Selects best error cases for publication
- Generates multiple report formats (CSV, Markdown, JSON)
- Easy to extend with new visualization types

Architecture:
- Modular design (selectors, reporters, generators)
- Plugin-based extensibility
- Clear separation of concerns
- Production-ready error handling

Usage:
    # Basic - process latest experiment
    python generate_all_test_visualizations.py

    # Specific experiment
    python generate_all_test_visualizations.py --experiment-dir results/optA_20251207_233941

    # Analysis only (skip visualization generation)
    python generate_all_test_visualizations.py --analysis-only

    # Custom selectors and reporters
    python generate_all_test_visualizations.py --selectors detection,classification --reporters csv,markdown
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our modular components
from scripts.visualization.selectors import (
    DetectionErrorSelector,
    ClassificationErrorSelector
)
from scripts.visualization.reporters import (
    CSVReporter,
    MarkdownReporter,
    JSONReporter
)
from scripts.visualization.generate_training_curves import TrainingCurvesGenerator


class VisualizationOrchestrator:
    """
    Unified orchestrator for all visualization tasks.

    Coordinates:
    - Detection visualization generation
    - Classification visualization generation
    - Training curves generation
    - Error case selection
    - Multi-format report generation

    Design: Modular, extensible, production-ready
    """

    def __init__(
        self,
        experiment_root: Path,
        output_dir: Optional[Path] = None,
        top_n: int = 5
    ):
        """
        Initialize orchestrator.

        Args:
            experiment_root: Root experiment directory (e.g., results/optA_20251207_233941)
            output_dir: Output directory for analysis results (default: {experiment_root}/visualization_summary)
            top_n: Number of top cases to select per category
        """
        self.experiment_root = Path(experiment_root)
        self.experiments_dir = self.experiment_root / "experiments"
        self.output_dir = output_dir or (self.experiment_root / "visualization_summary")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_n = top_n

        # Storage for results
        self.detection_results = []
        self.classification_results = []
        self.training_curves_info = {}
        self.summary_stats = {}

    def discover_experiments(self) -> List[Dict]:
        """
        Auto-discover experiment structure.

        Returns:
            List of dicts with experiment metadata
        """
        experiments = []

        if not self.experiments_dir.exists():
            print(f"[ERROR] Experiments directory not found: {self.experiments_dir}")
            return experiments

        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith('experiment_'):
                continue

            dataset_name = exp_dir.name.replace('experiment_', '')
            viz_dir = exp_dir / "visualizations"

            if not viz_dir.exists():
                continue

            # Count detection models
            detection_models = [d.name for d in viz_dir.iterdir()
                               if d.is_dir() and d.name.startswith('pred_detection_')]

            # Count classification models
            classification_models = [d.name for d in viz_dir.iterdir()
                                    if d.is_dir() and d.name.startswith('pred_classification_')]

            experiments.append({
                'dataset': dataset_name,
                'path': exp_dir,
                'viz_dir': viz_dir,
                'detection_models': detection_models,
                'classification_models': classification_models
            })

            print(f"[DISCOVERED] {dataset_name}: {len(detection_models)} det, {len(classification_models)} cls")

        return experiments

    def select_detection_errors(self, experiments: List[Dict]) -> pd.DataFrame:
        """
        Select interesting detection error cases from all experiments.

        Args:
            experiments: List of experiment metadata dicts

        Returns:
            DataFrame with selected detection errors
        """
        print("\n" + "="*80)
        print("SELECTING DETECTION ERROR CASES")
        print("="*80)

        selector = DetectionErrorSelector(top_n=self.top_n, include_perfect=True)
        all_selected = []

        for exp in experiments:
            print(f"\n[DATASET] {exp['dataset']}")

            # Find all detection metadata CSVs
            detection_csvs = list(exp['viz_dir'].glob("pred_detection_*/detection_metadata.csv"))

            for csv_path in detection_csvs:
                model_name = csv_path.parent.name.replace('pred_detection_', '')
                print(f"  [MODEL] {model_name}")

                try:
                    selected = selector.select_from_csv(csv_path)

                    if not selected.empty:
                        selected['dataset'] = exp['dataset']
                        selected['model'] = model_name
                        all_selected.append(selected)

                        # Print category stats
                        stats = selector.get_category_stats(pd.read_csv(csv_path))
                        print(f"    FP: {stats['fp_only']}, FN: {stats['fn_only']}, Mixed: {stats['mixed']}, Perfect: {stats['perfect']}")
                        print(f"    Selected: {len(selected)} cases")

                except Exception as e:
                    print(f"    [ERROR] {e}")

        # Combine all results
        if all_selected:
            result_df = pd.concat(all_selected, ignore_index=True)
            print(f"\n[TOTAL] Selected {len(result_df)} detection error cases")
            selector.print_summary()
            return result_df
        else:
            return pd.DataFrame()

    def select_classification_errors(self, experiments: List[Dict]) -> pd.DataFrame:
        """
        Select interesting classification error cases from all experiments.

        Args:
            experiments: List of experiment metadata dicts

        Returns:
            DataFrame with selected classification errors
        """
        print("\n" + "="*80)
        print("SELECTING CLASSIFICATION ERROR CASES")
        print("="*80)

        selector = ClassificationErrorSelector(top_n=self.top_n, conf_threshold=0.75, include_perfect=True)
        all_selected = []

        for exp in experiments:
            print(f"\n[DATASET] {exp['dataset']}")

            # Find all classification metadata CSVs
            classification_csvs = list(exp['viz_dir'].glob("pred_classification_*/classification_metadata_images.csv"))

            for csv_path in classification_csvs:
                model_name = csv_path.parent.name.replace('pred_classification_', '')
                print(f"  [MODEL] {model_name}")

                try:
                    selected = selector.select_from_csv(csv_path)

                    if not selected.empty:
                        selected['dataset'] = exp['dataset']
                        selected['model'] = model_name
                        all_selected.append(selected)

                        # Print category stats
                        stats = selector.get_category_stats(pd.read_csv(csv_path))
                        print(f"    All wrong: {stats['all_wrong']}, High conf errors: {stats['high_conf_errors']}")
                        print(f"    Mixed: {stats['mixed']}, Perfect: {stats['perfect']}")
                        print(f"    Selected: {len(selected)} cases")

                except Exception as e:
                    print(f"    [ERROR] {e}")

        # Combine all results
        if all_selected:
            result_df = pd.concat(all_selected, ignore_index=True)
            print(f"\n[TOTAL] Selected {len(result_df)} classification error cases")
            selector.print_summary()
            return result_df
        else:
            return pd.DataFrame()

    def generate_training_curves(self) -> Dict:
        """
        Generate training curves for all datasets.

        Returns:
            Dict with generation results
        """
        print("\n" + "="*80)
        print("GENERATING TRAINING CURVES")
        print("="*80)

        curves_output_dir = self.output_dir / "training_curves"

        try:
            generator = TrainingCurvesGenerator(
                experiment_dir=self.experiments_dir,
                output_dir=curves_output_dir,
                dpi=400
            )

            results = generator.generate_all(plot_types=['accuracy'])

            return {
                'success': True,
                'generated_figures': sum(results.values()),
                'output_dir': str(curves_output_dir),
                'plot_types': list(results.keys()),
                'results': results
            }

        except Exception as e:
            print(f"[ERROR] Failed to generate training curves: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def generate_reports(
        self,
        detection_errors: pd.DataFrame,
        classification_errors: pd.DataFrame
    ):
        """
        Generate multi-format reports.

        Args:
            detection_errors: DataFrame with selected detection errors
            classification_errors: DataFrame with selected classification errors
        """
        print("\n" + "="*80)
        print("GENERATING REPORTS")
        print("="*80)

        # Prepare summary data
        summary_data = {
            'summary': {
                'experiment_dir': str(self.experiment_root),
                'timestamp': datetime.now().isoformat(),
                'total_selected': len(detection_errors) + len(classification_errors),
                'detection_cases': len(detection_errors),
                'classification_cases': len(classification_errors)
            },
            'detection_errors': detection_errors,
            'classification_errors': classification_errors,
            'training_curves': self.training_curves_info
        }

        # CSV Reporter
        print("\n[CSV] Generating CSV reports...")
        csv_reporter = CSVReporter()
        csv_reporter.generate(detection_errors, self.output_dir / "selected_detection_errors.csv")
        csv_reporter.generate(classification_errors, self.output_dir / "selected_classification_errors.csv")

        # Combined CSV (like old selected_error_images.csv)
        combined = pd.concat([detection_errors, classification_errors], ignore_index=True)
        csv_reporter.generate(combined, self.output_dir / "selected_error_images.csv")

        # Markdown Reporter
        print("\n[MARKDOWN] Generating human-readable report...")
        md_reporter = MarkdownReporter(title="Visualization Analysis Report")
        md_reporter.generate(summary_data, self.output_dir / "visualization_report.md")

        # JSON Reporter
        print("\n[JSON] Generating machine-readable metadata...")
        json_reporter = JSONReporter(title="Visualization Metadata")
        json_reporter.generate(summary_data, self.output_dir / "visualization_metadata.json")

        print(f"\n[COMPLETE] All reports saved to: {self.output_dir}")

    def run_full_pipeline(self, skip_training_curves: bool = False):
        """
        Run complete visualization analysis pipeline.

        Args:
            skip_training_curves: Whether to skip training curves generation
        """
        print("\n" + "="*80)
        print("UNIFIED VISUALIZATION PIPELINE")
        print("="*80)
        print(f"Experiment: {self.experiment_root}")
        print(f"Output: {self.output_dir}")
        print(f"Top N per category: {self.top_n}")
        print("="*80)

        # Step 1: Discover experiments
        experiments = self.discover_experiments()
        if not experiments:
            print("\n[ERROR] No experiments found!")
            return

        print(f"\n[FOUND] {len(experiments)} datasets")

        # Step 2: Select detection errors
        detection_errors = self.select_detection_errors(experiments)

        # Step 3: Select classification errors
        classification_errors = self.select_classification_errors(experiments)

        # Step 4: Generate training curves
        if not skip_training_curves:
            self.training_curves_info = self.generate_training_curves()
        else:
            print("\n[SKIP] Training curves generation skipped")

        # Step 5: Generate reports
        self.generate_reports(detection_errors, classification_errors)

        # Final summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"Detection errors selected: {len(detection_errors)}")
        print(f"Classification errors selected: {len(classification_errors)}")
        print(f"Total selected cases: {len(detection_errors) + len(classification_errors)}")
        print(f"\nReports available in: {self.output_dir}")
        print("="*80)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Unified Visualization Orchestrator - Generate and analyze all visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process latest experiment
  python generate_all_test_visualizations.py

  # Specific experiment
  python generate_all_test_visualizations.py --experiment-dir results/optA_20251207_233941

  # Skip training curves (faster)
  python generate_all_test_visualizations.py --skip-training-curves

  # Custom output location
  python generate_all_test_visualizations.py --output custom_analysis
        """
    )

    parser.add_argument('--experiment-dir', type=str, default=None,
                       help='Experiment root directory (default: auto-detect latest)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for analysis (default: {experiment}/visualization_summary)')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Number of top cases per category (default: 5)')
    parser.add_argument('--skip-training-curves', action='store_true',
                       help='Skip training curves generation')

    args = parser.parse_args()

    # Auto-detect latest experiment if not specified
    if args.experiment_dir is None:
        results_dir = Path("results")
        if results_dir.exists():
            experiments = sorted([d for d in results_dir.iterdir()
                                if d.is_dir() and d.name.startswith('optA_')],
                               key=lambda x: x.name, reverse=True)
            if experiments:
                args.experiment_dir = str(experiments[0])
                print(f"[AUTO-DETECT] Using latest experiment: {args.experiment_dir}")
            else:
                print("[ERROR] No experiments found in results/")
                return 1
        else:
            print("[ERROR] results/ directory not found")
            return 1

    # Create orchestrator
    orchestrator = VisualizationOrchestrator(
        experiment_root=Path(args.experiment_dir),
        output_dir=Path(args.output) if args.output else None,
        top_n=args.top_n
    )

    # Run pipeline
    orchestrator.run_full_pipeline(skip_training_curves=args.skip_training_curves)

    return 0


if __name__ == '__main__':
    sys.exit(main())
