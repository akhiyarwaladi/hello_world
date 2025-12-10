#!/usr/bin/env python3
"""
CONSOLIDATE ALL VISUALIZATION OUTPUTS - ONE CENTRALIZED FOLDER

Problem: Outputs scattered across multiple locations (confusing!)
Solution: Copy/organize ALL outputs into ONE master folder

Output Structure:
    visualization_outputs/              ← EVERYTHING HERE!
    ├── 1_confusion_matrices/
    │   ├── individual/                # Per-model confusion matrices
    │   │   ├── iml_lifecycle_efficientnet_b1.png
    │   │   ├── iml_lifecycle_efficientnet_b0.png
    │   │   └── ... (24 files)
    │   └── consolidated/              # Publication 2x2 grids
    │       └── confusion_matrices_2x2.png
    │
    ├── 2_training_curves/
    │   ├── accuracy_iml_lifecycle.png
    │   ├── accuracy_md_2019_stages.png
    │   ├── accuracy_mp_idb_species.png
    │   └── accuracy_mp_idb_stages.png
    │
    ├── 3_test_visualizations/         # Selected error cases
    │   ├── detection_errors/
    │   │   ├── fp_PA171785.png       # False positives
    │   │   ├── fn_PA171699.png       # False negatives
    │   │   └── mixed_*.png           # Mixed errors
    │   └── classification_errors/
    │       ├── all_wrong_*.png
    │       └── high_conf_error_*.png
    │
    ├── 4_metadata/
    │   ├── selected_detection_errors.csv
    │   ├── selected_classification_errors.csv
    │   ├── selected_error_images.csv
    │   └── visualization_report.md
    │
    └── _README.txt                    # Quick guide

Usage:
    # Auto-detect and consolidate latest experiment
    python consolidate_all_outputs.py

    # Specific experiment
    python consolidate_all_outputs.py --experiment results/optA_20251207_233941

    # Custom output location
    python consolidate_all_outputs.py --output my_visualizations
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime


class VisualizationConsolidator:
    """Consolidate all visualization outputs into ONE folder."""

    def __init__(
        self,
        experiment_root: Path,
        output_dir: Path = Path("visualization_outputs")
    ):
        """
        Initialize consolidator.

        Args:
            experiment_root: Root experiment directory
            output_dir: Master output directory (default: visualization_outputs/)
        """
        self.experiment_root = Path(experiment_root)
        self.experiments_dir = self.experiment_root / "experiments"
        self.output_dir = Path(output_dir)

        # Create master structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cm_individual_dir = self.output_dir / "1_confusion_matrices" / "individual"
        self.cm_consolidated_dir = self.output_dir / "1_confusion_matrices" / "consolidated"
        self.curves_dir = self.output_dir / "2_training_curves"
        self.test_viz_dir = self.output_dir / "3_test_visualizations"
        self.metadata_dir = self.output_dir / "4_metadata"

        for d in [self.cm_individual_dir, self.cm_consolidated_dir,
                  self.curves_dir, self.test_viz_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'confusion_matrices': 0,
            'training_curves': 0,
            'test_visualizations': 0,
            'metadata_files': 0
        }

    def consolidate_confusion_matrices(self):
        """Copy all confusion matrices to consolidated location."""
        print("\n[1/4] Consolidating confusion matrices...")

        if not self.experiments_dir.exists():
            print("   [SKIP] Experiments directory not found")
            return

        # Individual per-model confusion matrices
        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith('experiment_'):
                continue

            dataset_name = exp_dir.name.replace('experiment_', '')

            # Find all classification model confusion matrices
            for cls_dir in exp_dir.glob("cls_*_focal"):
                cm_file = cls_dir / "confusion_matrix.png"
                if cm_file.exists():
                    model_name = cls_dir.name.replace('cls_', '').replace('_focal', '')
                    dest_name = f"{dataset_name}_{model_name}.png"
                    dest_path = self.cm_individual_dir / dest_name

                    shutil.copy2(cm_file, dest_path)
                    self.stats['confusion_matrices'] += 1
                    print(f"   ✓ {dest_name}")

        # Consolidated 2x2 grid
        consolidated_source = Path("luaran/auto_generated/figures/performance/confusion_matrices.png")
        if consolidated_source.exists():
            dest_path = self.cm_consolidated_dir / "confusion_matrices_2x2.png"
            shutil.copy2(consolidated_source, dest_path)
            self.stats['confusion_matrices'] += 1
            print(f"   ✓ confusion_matrices_2x2.png (consolidated)")
        else:
            print(f"   [INFO] Consolidated CM not found, generating...")
            # Try to generate
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable,
                    "scripts/visualization/generate_consolidated_confusion_matrices.py",
                    "--experiment-dir", str(self.experiments_dir)
                ], capture_output=True, text=True)

                if result.returncode == 0 and consolidated_source.exists():
                    dest_path = self.cm_consolidated_dir / "confusion_matrices_2x2.png"
                    shutil.copy2(consolidated_source, dest_path)
                    self.stats['confusion_matrices'] += 1
                    print(f"   ✓ confusion_matrices_2x2.png (generated)")
            except Exception as e:
                print(f"   [WARNING] Could not generate consolidated CM: {e}")

        print(f"   [TOTAL] {self.stats['confusion_matrices']} confusion matrices copied")

    def consolidate_training_curves(self):
        """Copy all training curves to consolidated location."""
        print("\n[2/4] Consolidating training curves...")

        source_dir = Path("luaran/auto_generated/figures/training_curves")

        if not source_dir.exists():
            print(f"   [INFO] Training curves not found, generating...")
            # Try to generate
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable,
                    "scripts/visualization/generate_training_curves.py",
                    "--experiment-dir", str(self.experiments_dir),
                    "--output-dir", str(source_dir)
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"   [SKIP] Could not generate training curves")
                    return
            except Exception as e:
                print(f"   [SKIP] Error generating curves: {e}")
                return

        # Copy all PNG files
        for curve_file in source_dir.glob("*.png"):
            dest_path = self.curves_dir / curve_file.name
            shutil.copy2(curve_file, dest_path)
            self.stats['training_curves'] += 1
            print(f"   ✓ {curve_file.name}")

        print(f"   [TOTAL] {self.stats['training_curves']} training curves copied")

    def consolidate_test_visualizations(self):
        """Copy selected test visualizations to consolidated location."""
        print("\n[3/4] Consolidating test visualizations...")

        # Read selected error cases from visualization_summary
        summary_dir = self.experiment_root / "visualization_summary"

        if not summary_dir.exists():
            print(f"   [INFO] No visualization_summary found, generating...")
            # Try to generate
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable,
                    "scripts/visualization/generate_all_test_visualizations.py",
                    "--experiment-dir", str(self.experiment_root),
                    "--skip-training-curves"
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"   [SKIP] Could not generate test visualizations")
                    return
            except Exception as e:
                print(f"   [SKIP] Error: {e}")
                return

        # Copy selected test images
        detection_csv = summary_dir / "selected_detection_errors.csv"
        classification_csv = summary_dir / "selected_classification_errors.csv"

        # Copy detection error images
        if detection_csv.exists():
            import pandas as pd
            df = pd.read_csv(detection_csv)

            det_output = self.test_viz_dir / "detection_errors"
            det_output.mkdir(exist_ok=True)

            for _, row in df.head(20).iterrows():  # Top 20 cases
                src_path = Path(row['image_file'])
                if src_path.exists():
                    category = row['error_category'].replace(' ', '_').replace('(', '').replace(')', '').lower()
                    dest_name = f"{category}_{row['image_name']}.png"
                    dest_path = det_output / dest_name
                    shutil.copy2(src_path, dest_path)
                    self.stats['test_visualizations'] += 1

            print(f"   ✓ {self.stats['test_visualizations']} detection error images")

        # Copy classification error images
        if classification_csv.exists():
            import pandas as pd
            df = pd.read_csv(classification_csv)

            cls_output = self.test_viz_dir / "classification_errors"
            cls_output.mkdir(exist_ok=True)

            count_before = self.stats['test_visualizations']
            for _, row in df.head(20).iterrows():  # Top 20 cases
                src_path = Path(row['image_file'])
                if src_path.exists():
                    category = row['error_category'].replace(' ', '_').replace('(', '').replace(')', '').replace('>', '').lower()
                    dest_name = f"{category}_{row['image_name']}.png"
                    dest_path = cls_output / dest_name
                    shutil.copy2(src_path, dest_path)
                    self.stats['test_visualizations'] += 1

            print(f"   ✓ {self.stats['test_visualizations'] - count_before} classification error images")

        print(f"   [TOTAL] {self.stats['test_visualizations']} test visualizations copied")

    def consolidate_metadata(self):
        """Copy all metadata and reports to consolidated location."""
        print("\n[4/4] Consolidating metadata...")

        summary_dir = self.experiment_root / "visualization_summary"

        if not summary_dir.exists():
            print("   [SKIP] No metadata found")
            return

        # Copy all CSV and MD files
        for file_path in summary_dir.glob("*"):
            if file_path.suffix in ['.csv', '.md', '.json']:
                dest_path = self.metadata_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                self.stats['metadata_files'] += 1
                print(f"   ✓ {file_path.name}")

        print(f"   [TOTAL] {self.stats['metadata_files']} metadata files copied")

    def create_readme(self):
        """Create README in output directory."""
        readme_path = self.output_dir / "_README.txt"

        content = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   CONSOLIDATED VISUALIZATION OUTPUTS                         ║
║                          ONE FOLDER - EVERYTHING                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: {self.experiment_root}

📂 FOLDER STRUCTURE:

1_confusion_matrices/
├── individual/          {self.stats['confusion_matrices']-1} per-model confusion matrices
└── consolidated/        1 publication 2x2 grid

2_training_curves/       {self.stats['training_curves']} accuracy curves (1 per dataset)

3_test_visualizations/   {self.stats['test_visualizations']} selected error case images
├── detection_errors/    Top 20 detection errors (FP, FN, Mixed)
└── classification_errors/ Top 20 classification errors

4_metadata/              {self.stats['metadata_files']} CSV/Markdown/JSON files
├── selected_detection_errors.csv
├── selected_classification_errors.csv
├── selected_error_images.csv
└── visualization_report.md

═══════════════════════════════════════════════════════════════════════════════

📊 QUICK ACCESS:

FOR PAPER/PUBLICATION:
→ 1_confusion_matrices/consolidated/confusion_matrices_2x2.png
→ 2_training_curves/*.png (all 4 curves)
→ 3_test_visualizations/ (selected error examples)

FOR DETAILED ANALYSIS:
→ 1_confusion_matrices/individual/ (per-model performance)
→ 4_metadata/*.csv (sortable data)
→ 4_metadata/visualization_report.md (human-readable summary)

═══════════════════════════════════════════════════════════════════════════════

💡 TIPS:

1. Open CSVs in Excel and sort by:
   - paper_score (descending) for best cases
   - error_category (group by type)
   - confidence (to find interesting failures)

2. Use visualization_report.md for quick overview

3. All image paths are relative - easy to share folder

═══════════════════════════════════════════════════════════════════════════════

🔄 TO REGENERATE:

python scripts/visualization/consolidate_all_outputs.py

"""

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"\n   ✓ Created README: {readme_path}")

    def run(self):
        """Run complete consolidation."""
        print("="*80)
        print("CONSOLIDATING ALL VISUALIZATION OUTPUTS")
        print("="*80)
        print(f"Source: {self.experiment_root}")
        print(f"Output: {self.output_dir}")
        print("="*80)

        self.consolidate_confusion_matrices()
        self.consolidate_training_curves()
        self.consolidate_test_visualizations()
        self.consolidate_metadata()
        self.create_readme()

        print("\n" + "="*80)
        print("CONSOLIDATION COMPLETE")
        print("="*80)
        print(f"✓ Confusion matrices: {self.stats['confusion_matrices']}")
        print(f"✓ Training curves: {self.stats['training_curves']}")
        print(f"✓ Test visualizations: {self.stats['test_visualizations']}")
        print(f"✓ Metadata files: {self.stats['metadata_files']}")
        print(f"\n📁 ALL OUTPUTS IN: {self.output_dir.absolute()}")
        print(f"📄 READ: {self.output_dir.absolute()}/_README.txt")
        print("="*80)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Consolidate ALL visualization outputs into ONE folder'
    )
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment directory (default: auto-detect latest)')
    parser.add_argument('--output', type=str, default='visualization_outputs',
                       help='Output directory (default: visualization_outputs/)')

    args = parser.parse_args()

    # Auto-detect latest experiment
    if args.experiment is None:
        results_dir = Path("results")
        if results_dir.exists():
            experiments = sorted([d for d in results_dir.iterdir()
                                if d.is_dir() and d.name.startswith('optA_')],
                               key=lambda x: x.name, reverse=True)
            if experiments:
                args.experiment = str(experiments[0])
                print(f"[AUTO-DETECT] Using latest: {args.experiment}")
            else:
                print("[ERROR] No experiments found")
                return 1
        else:
            print("[ERROR] results/ directory not found")
            return 1

    # Run consolidation
    consolidator = VisualizationConsolidator(
        experiment_root=Path(args.experiment),
        output_dir=Path(args.output)
    )

    consolidator.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
