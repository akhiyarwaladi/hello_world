#!/usr/bin/env python3
"""
UNIFIED VISUALIZATION GENERATOR - CENTRALIZED OUTPUT
====================================================

Generate SEMUA visualizations LANGSUNG ke SATU FOLDER TERPUSAT.
NO copying, NO scattered outputs - everything generated to ONE place from the start!

Output Structure:
    visualization_outputs/              ← SATU FOLDER UNTUK SEMUA!
    ├── confusion_matrices/
    │   ├── individual/                # Per-model CMs (auto-generated during training)
    │   └── consolidated_2x2.png       # 2x2 grid publication figure
    │
    ├── training_curves/
    │   ├── accuracy_iml_lifecycle.png
    │   ├── accuracy_md_2019_stages.png
    │   ├── accuracy_mp_idb_species.png
    │   └── accuracy_mp_idb_stages.png
    │
    ├── test_visualizations/
    │   └── selected_cases/            # Top error cases only (not all images)
    │       ├── detection/
    │       └── classification/
    │
    ├── metadata/
    │   ├── selected_detection_errors.csv
    │   ├── selected_classification_errors.csv
    │   ├── selected_error_images.csv
    │   └── visualization_report.md
    │
    └── _README.txt                    # Guide

Usage:
    # Generate everything to default location (visualization_outputs/)
    python generate_all_centralized.py

    # Custom output directory
    python generate_all_centralized.py --output my_viz

    # Specific experiment
    python generate_all_centralized.py --experiment results/optA_20251207_233941
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import shutil

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.visualization.selectors import DetectionErrorSelector, ClassificationErrorSelector
from scripts.visualization.reporters import CSVReporter, MarkdownReporter, JSONReporter
from scripts.visualization.generate_training_curves import TrainingCurvesGenerator
# REMOVED: Consolidated confusion matrix no longer needed
# from scripts.visualization.generate_consolidated_confusion_matrices import ConsolidatedConfusionMatrixGenerator


class CentralizedVisualizationGenerator:
    """
    Generate ALL visualizations directly to ONE centralized folder.

    No scattered outputs, no manual copying - everything generated in place!
    """

    def __init__(
        self,
        experiment_root: Path,
        output_dir: Path = Path("visualization_outputs"),
        top_n: int = 20  # More cases for flexibility
    ):
        """
        Initialize generator.

        Args:
            experiment_root: Experiment root (e.g., results/optA_20251207_233941)
            output_dir: Master output directory
            top_n: Number of cases per category for selection
        """
        self.experiment_root = Path(experiment_root)
        self.experiments_dir = self.experiment_root / "experiments"
        self.output_dir = Path(output_dir)
        self.top_n = top_n

        # Create centralized structure
        self.cm_dir = self.output_dir / "confusion_matrices"
        self.cm_individual_dir = self.cm_dir / "individual"
        self.curves_dir = self.output_dir / "training_curves"
        self.test_viz_dir = self.output_dir / "test_visualizations" / "selected_cases"
        self.metadata_dir = self.output_dir / "metadata"

        for d in [self.cm_dir, self.cm_individual_dir, self.curves_dir,
                  self.test_viz_dir / "detection", self.test_viz_dir / "classification",
                  self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.results = {}

    def discover_experiments(self) -> List[Dict]:
        """Auto-discover experiment structure."""
        experiments = []

        if not self.experiments_dir.exists():
            print(f"[ERROR] Experiments directory not found: {self.experiments_dir}")
            return experiments

        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith('experiment_'):
                continue

            dataset_name = exp_dir.name.replace('experiment_', '')
            viz_dir = exp_dir / "visualizations"

            detection_models = [d.name for d in viz_dir.iterdir()
                               if d.is_dir() and d.name.startswith('pred_detection_')] if viz_dir.exists() else []

            classification_models = [d.name for d in viz_dir.iterdir()
                                    if d.is_dir() and d.name.startswith('pred_classification_')] if viz_dir.exists() else []

            experiments.append({
                'dataset': dataset_name,
                'path': exp_dir,
                'viz_dir': viz_dir,
                'detection_models': detection_models,
                'classification_models': classification_models
            })

            print(f"[DISCOVERED] {dataset_name}: {len(detection_models)} det, {len(classification_models)} cls")

        return experiments

    def generate_confusion_matrices(self):
        """Generate confusion matrices DIRECTLY to centralized folder."""
        print("\n" + "="*80)
        print("[1/4] GENERATING CONFUSION MATRICES")
        print("="*80)

        # Clean confusion matrices folder before regenerating
        print("\n[1.-1] Cleaning confusion matrices folder...")
        if self.cm_individual_dir.exists():
            for f in self.cm_individual_dir.glob("*.png"):
                f.unlink()
        if self.cm_dir.exists():
            for f in self.cm_dir.glob("*.png"):
                f.unlink()
        print("   ✓ Cleaned old confusion matrices")

        # Step 0: Regenerate publication-quality confusion matrices
        # Features: NO 45-degree rotation, dataset-specific colors, dual annotations
        print("\n[1.0] Regenerating publication-quality confusion matrices...")
        print("        (NO rotation, dataset-specific colormaps)")
        try:
            import subprocess
            result = subprocess.run([
                sys.executable,
                "scripts/visualization/regenerate_publication_quality_confusion_matrices.py"
            ], capture_output=True, text=True, timeout=900)
            if result.returncode == 0:
                print("   ✓ Successfully regenerated all confusion matrices")
            else:
                print(f"   [WARNING] Some confusion matrices failed to regenerate")
                if result.stderr:
                    print(f"   {result.stderr[:200]}")
        except Exception as e:
            print(f"   [WARNING] Could not regenerate CMs: {e}")

        # Step 1: Copy individual confusion matrices (now with beautiful styling)
        print("\n[1.1] Collecting beautiful confusion matrices...")
        count = 0

        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith('experiment_'):
                continue

            dataset_name = exp_dir.name.replace('experiment_', '')

            for cls_dir in exp_dir.glob("cls_*_focal"):
                cm_file = cls_dir / "confusion_matrix.png"
                if cm_file.exists():
                    model_name = cls_dir.name.replace('cls_', '').replace('_focal', '')
                    dest_name = f"{dataset_name}_{model_name}.png"
                    dest_path = self.cm_individual_dir / dest_name
                    shutil.copy2(cm_file, dest_path)
                    count += 1

        print(f"   ✓ Collected {count} individual confusion matrices")

        # REMOVED: Consolidated 2x2 grid generation (not needed)
        # User only wants individual confusion matrices
        self.results['confusion_matrices'] = count
        print(f"\n[1.2] Skipped: Consolidated confusion matrix (individual matrices only)")

    def generate_training_curves(self):
        """Generate training curves DIRECTLY to centralized folder."""
        print("\n" + "="*80)
        print("[2/4] GENERATING TRAINING CURVES")
        print("="*80)

        # Clean training curves folder before regenerating
        print("\n[2.0] Cleaning training curves folder...")
        if self.curves_dir.exists():
            for f in self.curves_dir.glob("*.png"):
                f.unlink()
            print("   ✓ Cleaned old training curves")

        try:
            generator = TrainingCurvesGenerator(
                experiment_dir=self.experiments_dir,
                output_dir=self.curves_dir,  # Direct to centralized folder!
                dpi=400
            )

            # Generate BOTH accuracy and loss curves (complete set)
            results = generator.generate_all(plot_types=['accuracy', 'loss'])
            self.results['training_curves'] = sum(results.values())

        except Exception as e:
            print(f"   [ERROR] Failed to generate training curves: {e}")
            self.results['training_curves'] = 0

    def select_and_copy_error_cases(self):
        """Select error cases and copy to centralized folder."""
        print("\n" + "="*80)
        print("[3/4] SELECTING & COPYING ERROR CASES")
        print("="*80)

        # Clean test visualizations folder before regenerating
        print("\n[3.0] Cleaning test visualizations folder...")
        if self.test_viz_dir.exists():
            import shutil
            shutil.rmtree(self.test_viz_dir)
        # Recreate folder structure
        (self.test_viz_dir / "detection").mkdir(parents=True, exist_ok=True)
        (self.test_viz_dir / "classification").mkdir(parents=True, exist_ok=True)
        print("   ✓ Cleaned old test visualizations")

        experiments = self.discover_experiments()

        # Detection errors
        print("\n[3.1] Selecting detection error cases...")
        selector = DetectionErrorSelector(top_n=self.top_n, include_perfect=False)
        detection_selected = []

        for exp in experiments:
            detection_csvs = list(exp['viz_dir'].glob("pred_detection_*/detection_metadata.csv"))

            for csv_path in detection_csvs:
                try:
                    selected = selector.select_from_csv(csv_path)
                    if not selected.empty:
                        selected['dataset'] = exp['dataset']
                        selected['model'] = csv_path.parent.name.replace('pred_detection_', '')
                        detection_selected.append(selected)
                except Exception as e:
                    print(f"   [WARNING] {csv_path}: {e}")

        if detection_selected:
            import pandas as pd
            detection_df = pd.concat(detection_selected, ignore_index=True)

            # Copy top cases to centralized folder
            for _, row in detection_df.head(self.top_n).iterrows():
                src = Path(row['image_file'])
                if src.exists():
                    category = row['error_category'].replace(' ', '_').replace('(', '').replace(')', '').lower()
                    dest_name = f"{category}_{row['dataset']}_{row['image_name']}.png"
                    dest_path = self.test_viz_dir / "detection" / dest_name
                    shutil.copy2(src, dest_path)

            print(f"   ✓ Selected {len(detection_df)} detection cases, copied top {self.top_n}")
            self.results['detection_cases'] = len(detection_df)
        else:
            print("   [WARNING] No detection cases found")
            detection_df = pd.DataFrame()
            self.results['detection_cases'] = 0

        # Classification errors
        print("\n[3.2] Selecting classification error cases...")
        selector = ClassificationErrorSelector(top_n=self.top_n, include_perfect=False)
        classification_selected = []

        for exp in experiments:
            classification_csvs = list(exp['viz_dir'].glob("pred_classification_*/classification_metadata_images.csv"))

            for csv_path in classification_csvs:
                try:
                    selected = selector.select_from_csv(csv_path)
                    if not selected.empty:
                        selected['dataset'] = exp['dataset']
                        selected['model'] = csv_path.parent.name.replace('pred_classification_', '')
                        classification_selected.append(selected)
                except Exception as e:
                    print(f"   [WARNING] {csv_path}: {e}")

        if classification_selected:
            import pandas as pd
            classification_df = pd.concat(classification_selected, ignore_index=True)

            # Copy top cases to centralized folder
            for _, row in classification_df.head(self.top_n).iterrows():
                src = Path(row['image_file'])
                if src.exists():
                    category = row['error_category'].replace(' ', '_').replace('(', '').replace(')', '').replace('>', '').lower()
                    dest_name = f"{category}_{row['dataset']}_{row['image_name']}.png"
                    dest_path = self.test_viz_dir / "classification" / dest_name
                    shutil.copy2(src, dest_path)

            print(f"   ✓ Selected {len(classification_df)} classification cases, copied top {self.top_n}")
            self.results['classification_cases'] = len(classification_df)
        else:
            print("   [WARNING] No classification cases found")
            classification_df = pd.DataFrame()
            self.results['classification_cases'] = 0

        # Save metadata
        csv_reporter = CSVReporter()
        csv_reporter.generate(detection_df, self.metadata_dir / "selected_detection_errors.csv")
        csv_reporter.generate(classification_df, self.metadata_dir / "selected_classification_errors.csv")

        # Combined
        combined = pd.concat([detection_df, classification_df], ignore_index=True)
        csv_reporter.generate(combined, self.metadata_dir / "selected_error_images.csv")

        # Markdown report
        md_reporter = MarkdownReporter()
        summary_data = {
            'summary': {
                'experiment': str(self.experiment_root),
                'timestamp': datetime.now().isoformat(),
                'detection_cases': len(detection_df),
                'classification_cases': len(classification_df)
            },
            'detection_errors': detection_df,
            'classification_errors': classification_df
        }
        md_reporter.generate(summary_data, self.metadata_dir / "visualization_report.md")

    def create_readme(self):
        """Create README guide."""
        print("\n" + "="*80)
        print("[4/4] CREATING README")
        print("="*80)

        readme_path = self.output_dir / "_README.txt"

        content = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CENTRALIZED VISUALIZATION OUTPUTS - ONE FOLDER!                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source Experiment: {self.experiment_root}
Output Directory: {self.output_dir.absolute()}

═══════════════════════════════════════════════════════════════════════════════

📂 FOLDER STRUCTURE (EVERYTHING IN ONE PLACE!):

confusion_matrices/
├── individual/              {self.results.get('confusion_matrices', 0)} per-model confusion matrices
└── consolidated_2x2.png     Publication 2x2 grid (best models)

training_curves/             {self.results.get('training_curves', 0)} accuracy curves
├── accuracy_iml_lifecycle.png
├── accuracy_md_2019_stages.png
├── accuracy_mp_idb_species.png
└── accuracy_mp_idb_stages.png

test_visualizations/
└── selected_cases/          Top {self.top_n} error cases each type
    ├── detection/           Detection errors (FP, FN, Mixed)
    └── classification/      Classification errors

metadata/                    Analysis data & reports
├── selected_detection_errors.csv      ({self.results.get('detection_cases', 0)} cases)
├── selected_classification_errors.csv ({self.results.get('classification_cases', 0)} cases)
├── selected_error_images.csv          (combined)
└── visualization_report.md            (human-readable)

═══════════════════════════════════════════════════════════════════════════════

🎯 QUICK ACCESS FOR PAPER/LAPORAN:

📊 FIGURES:
   → confusion_matrices/consolidated_2x2.png    (Publication figure)
   → training_curves/*.png                      (4 accuracy curves)
   → test_visualizations/selected_cases/        (Error examples)

📋 DATA/ANALYSIS:
   → metadata/selected_*_errors.csv             (Sortable in Excel)
   → metadata/visualization_report.md           (Summary)

═══════════════════════════════════════════════════════════════════════════════

💡 CARA PAKAI:

1. UNTUK PAPER:
   - Ambil: confusion_matrices/consolidated_2x2.png
   - Ambil: training_curves/*.png (pilih yang perlu)
   - Ambil: test_visualizations/selected_cases/ (error examples)

2. UNTUK ANALISIS:
   - Buka: metadata/selected_detection_errors.csv di Excel
   - Sort by: paper_score (descending) atau error_category
   - Lihat image_file column untuk lokasi gambar

3. UNTUK LAPORAN:
   - Baca: metadata/visualization_report.md
   - Quick summary semua results

═══════════════════════════════════════════════════════════════════════════════

🔄 TO REGENERATE:

python scripts/visualization/generate_all_centralized.py

═══════════════════════════════════════════════════════════════════════════════
"""

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"   ✓ Created: _README.txt")

    def run(
        self,
        generate_confusion: bool = True,
        generate_curves: bool = True,
        generate_bbox: bool = True
    ):
        """
        Run centralized generation with selective options.

        Args:
            generate_confusion: Generate confusion matrices
            generate_curves: Generate training curves
            generate_bbox: Generate bbox predictions (error cases)
        """
        print("\n" + "="*80)
        print("CENTRALIZED VISUALIZATION GENERATION")
        print("="*80)
        print(f"Experiment: {self.experiment_root}")
        print(f"Output: {self.output_dir.absolute()}")
        print(f"Top N per category: {self.top_n}")
        print()
        print("Generation mode:")
        print(f"  Confusion matrices: {'✓' if generate_confusion else '✗'}")
        print(f"  Training curves: {'✓' if generate_curves else '✗'}")
        print(f"  Bbox predictions: {'✓' if generate_bbox else '✗'}")
        print("="*80)

        # Selective generation
        if generate_confusion:
            self.generate_confusion_matrices()
        else:
            print("\n[SKIPPED] Confusion matrices generation")

        if generate_curves:
            self.generate_training_curves()
        else:
            print("\n[SKIPPED] Training curves generation")

        if generate_bbox:
            self.select_and_copy_error_cases()
        else:
            print("\n[SKIPPED] Bbox predictions generation")

        # Always create README
        self.create_readme()

        # Summary
        print("\n" + "="*80)
        print("✅ GENERATION COMPLETE - EVERYTHING IN ONE PLACE!")
        print("="*80)
        if generate_confusion:
            print(f"📊 Confusion matrices: {self.results.get('confusion_matrices', 0)}")
        if generate_curves:
            print(f"📈 Training curves: {self.results.get('training_curves', 0)}")
        if generate_bbox:
            print(f"🔍 Detection cases: {self.results.get('detection_cases', 0)}")
            print(f"🔬 Classification cases: {self.results.get('classification_cases', 0)}")
        print(f"\n📁 LOCATION: {self.output_dir.absolute()}")
        print(f"📄 READ: {(self.output_dir / '_README.txt').absolute()}")
        print("="*80)


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(
        description='Generate ALL visualizations to ONE centralized folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate everything (default)
  python generate_all_centralized.py

  # Generate only confusion matrices
  python generate_all_centralized.py --confusion-matrices-only

  # Generate only training curves
  python generate_all_centralized.py --training-curves-only

  # Generate only bbox predictions
  python generate_all_centralized.py --bbox-predictions-only

  # Generate confusion matrices + training curves
  python generate_all_centralized.py --confusion-matrices --training-curves
        """
    )
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment directory (default: auto-detect latest)')
    parser.add_argument('--output', type=str, default='visualization_outputs',
                       help='Output directory (default: visualization_outputs/)')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of cases per category (default: 20)')

    # Selective generation flags
    parser.add_argument('--confusion-matrices', action='store_true',
                       help='Generate only confusion matrices')
    parser.add_argument('--training-curves', action='store_true',
                       help='Generate only training curves')
    parser.add_argument('--bbox-predictions', action='store_true',
                       help='Generate only bbox predictions (error cases)')

    # Convenience shortcuts (mutually exclusive with individual flags)
    parser.add_argument('--confusion-matrices-only', action='store_true',
                       help='Shortcut: generate ONLY confusion matrices')
    parser.add_argument('--training-curves-only', action='store_true',
                       help='Shortcut: generate ONLY training curves')
    parser.add_argument('--bbox-predictions-only', action='store_true',
                       help='Shortcut: generate ONLY bbox predictions')

    args = parser.parse_args()

    # Determine what to generate
    generate_all = not any([
        args.confusion_matrices, args.training_curves, args.bbox_predictions,
        args.confusion_matrices_only, args.training_curves_only, args.bbox_predictions_only
    ])

    # Handle shortcuts
    if args.confusion_matrices_only:
        args.confusion_matrices = True
        generate_all = False
    if args.training_curves_only:
        args.training_curves = True
        generate_all = False
    if args.bbox_predictions_only:
        args.bbox_predictions = True
        generate_all = False

    # Auto-detect
    if args.experiment is None:
        results_dir = Path("results")
        if results_dir.exists():
            experiments = sorted([d for d in results_dir.iterdir()
                                if d.is_dir() and d.name.startswith('optA_')],
                               key=lambda x: x.name, reverse=True)
            if experiments:
                args.experiment = str(experiments[0])
                print(f"[AUTO-DETECT] Latest: {args.experiment}")
            else:
                print("[ERROR] No experiments found")
                return 1
        else:
            print("[ERROR] results/ not found")
            return 1

    # Print selective generation info
    if not generate_all:
        print("\n[SELECTIVE GENERATION MODE]")
        if args.confusion_matrices:
            print("  ✓ Confusion matrices")
        if args.training_curves:
            print("  ✓ Training curves")
        if args.bbox_predictions:
            print("  ✓ Bbox predictions")
        print()

    # Generate
    generator = CentralizedVisualizationGenerator(
        experiment_root=Path(args.experiment),
        output_dir=Path(args.output),
        top_n=args.top_n
    )

    generator.run(
        generate_confusion=generate_all or args.confusion_matrices,
        generate_curves=generate_all or args.training_curves,
        generate_bbox=generate_all or args.bbox_predictions
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
