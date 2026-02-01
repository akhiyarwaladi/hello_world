#!/usr/bin/env python3
"""
UNIFIED FIGURE GENERATOR - ALL PAPER FIGURES IN ONE COMMAND
============================================================

Generates ALL 6 paper figures + metadata to a single centralized folder.

Paper Figure Mapping:
    Figure 1 → Augmentation examples (4 datasets, 2x2 grid)
    Figure 2 → Pipeline architecture diagram
    Figure 3 → Confusion matrices (4 best models, individual PNGs)
    Figure 4 → Training accuracy curves (4 datasets, best vs worst)
    Figure 5 → Detection error patterns (selected from experiment)
    Figure 6 → Classification error patterns (selected from experiment)

Output Structure:
    visualization_outputs/
    ├── fig1_augmentation/
    │   └── augmentation_4datasets_combined_2x2.png
    ├── fig2_pipeline/
    │   └── pipeline_architecture_horizontal.png
    ├── fig3_confusion_matrices/
    │   └── individual/          (per-model CMs)
    ├── fig4_training_curves/
    │   ├── accuracy_iml_lifecycle.png
    │   ├── accuracy_mp_idb_species.png
    │   ├── accuracy_mp_idb_stages.png
    │   └── accuracy_md_2019_stages.png
    ├── fig5_detection_errors/
    │   └── (selected detection error images)
    ├── fig6_classification_errors/
    │   └── (selected classification error images)
    ├── metadata/
    │   ├── selected_detection_errors.csv
    │   ├── selected_classification_errors.csv
    │   └── visualization_report.md
    └── _README.txt

Usage:
    # Generate ALL figures (default)
    python scripts/visualization/generate_all_centralized.py

    # Custom experiment
    python scripts/visualization/generate_all_centralized.py --experiment results/optA_20251016_200330

    # Selective generation (by figure number)
    python scripts/visualization/generate_all_centralized.py --only fig1 fig3 fig4
    python scripts/visualization/generate_all_centralized.py --only fig5 fig6
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import shutil


def sanitize_filename(name: str) -> str:
    """Remove/replace characters invalid for Windows filenames."""
    return re.sub(r'[<>:"/\\|?*\s]+', '_', name).strip('_')

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


# Figure-to-folder mapping (single source of truth)
FIGURE_DIRS = {
    'fig1': 'fig1_augmentation',
    'fig2': 'fig2_pipeline',
    'fig3': 'fig3_confusion_matrices',
    'fig4': 'fig4_training_curves',
    'fig5': 'fig5_detection_errors',
    'fig6': 'fig6_classification_errors',
}

FIGURE_LABELS = {
    'fig1': 'Figure 1 - Augmentation Examples',
    'fig2': 'Figure 2 - Pipeline Architecture',
    'fig3': 'Figure 3 - Confusion Matrices',
    'fig4': 'Figure 4 - Training Curves',
    'fig5': 'Figure 5 - Detection Errors',
    'fig6': 'Figure 6 - Classification Errors',
}


class CentralizedVisualizationGenerator:
    """Generate ALL paper figures to ONE centralized folder."""

    def __init__(
        self,
        experiment_root: Path,
        output_dir: Path = Path("visualization_outputs"),
        top_n: int = 20
    ):
        self.experiment_root = Path(experiment_root)
        self.experiments_dir = self.experiment_root / "experiments"
        self.output_dir = Path(output_dir)
        self.top_n = top_n

        # Create figure directories
        self.fig_dirs = {}
        for key, dirname in FIGURE_DIRS.items():
            d = self.output_dir / dirname
            d.mkdir(parents=True, exist_ok=True)
            self.fig_dirs[key] = d

        # Subdirectories
        (self.fig_dirs['fig3'] / "individual").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)

        self.results = {}

    def discover_experiments(self) -> List[Dict]:
        """Auto-discover experiment structure."""
        experiments = []

        if not self.experiments_dir.exists():
            print(f"  [ERROR] Experiments directory not found: {self.experiments_dir}")
            return experiments

        for exp_dir in sorted(self.experiments_dir.iterdir()):
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
            print(f"  [DISCOVERED] {dataset_name}: {len(detection_models)} det, {len(classification_models)} cls")

        return experiments

    # ------------------------------------------------------------------
    # FIGURE 1: Augmentation Examples
    # ------------------------------------------------------------------
    def generate_fig1_augmentation(self):
        """Figure 1 - Medical-safe augmentation examples across 4 datasets."""
        print("\n" + "="*80)
        print("[Fig 1/6] AUGMENTATION EXAMPLES")
        print("="*80)

        output_path = self.fig_dirs['fig1'] / "augmentation_4datasets_combined_2x2.png"
        script_path = project_root / "scripts" / "visualization" / "generate_combined_4datasets_augmentation.py"

        if not script_path.exists():
            print(f"  [ERROR] Script not found: {script_path}")
            self.results['fig1'] = 0
            return

        # Clean existing
        for f in self.fig_dirs['fig1'].glob("*.png"):
            f.unlink()

        try:
            result = subprocess.run([
                sys.executable, str(script_path),
                "--output", str(output_path),
                "--dpi", "300"
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and output_path.exists():
                size_kb = output_path.stat().st_size / 1024
                print(f"  [OK] Saved: {output_path.name} ({size_kb:.0f} KB)")
                self.results['fig1'] = 1
            else:
                print(f"  [ERROR] Generation failed")
                if result.stderr:
                    print(f"  {result.stderr[:300]}")
                self.results['fig1'] = 0
        except Exception as e:
            print(f"  [ERROR] {e}")
            self.results['fig1'] = 0

    # ------------------------------------------------------------------
    # FIGURE 2: Pipeline Architecture Diagram
    # ------------------------------------------------------------------
    def generate_fig2_pipeline(self):
        """Figure 2 - System architecture overview diagram."""
        print("\n" + "="*80)
        print("[Fig 2/6] PIPELINE ARCHITECTURE DIAGRAM")
        print("="*80)

        script_path = project_root / "scripts" / "visualization" / "generate_pipeline_architecture_diagram.py"

        if not script_path.exists():
            print(f"  [ERROR] Script not found: {script_path}")
            self.results['fig2'] = 0
            return

        # Clean existing
        for f in self.fig_dirs['fig2'].glob("*.png"):
            f.unlink()

        # The pipeline script saves to luaran/auto_generated/..., we run it
        # then copy the result to our centralized folder
        try:
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, timeout=120)

            # Copy from its default output location to centralized folder
            default_output = project_root / "luaran" / "auto_generated" / "figures" / "pipeline_diagrams" / "pipeline_architecture_horizontal.png"
            dest_path = self.fig_dirs['fig2'] / "pipeline_architecture_horizontal.png"

            if default_output.exists():
                shutil.copy2(default_output, dest_path)
                size_kb = dest_path.stat().st_size / 1024
                print(f"  [OK] Saved: {dest_path.name} ({size_kb:.0f} KB)")
                self.results['fig2'] = 1
            else:
                print(f"  [ERROR] Output not found at {default_output}")
                self.results['fig2'] = 0
        except Exception as e:
            print(f"  [ERROR] {e}")
            self.results['fig2'] = 0

    # ------------------------------------------------------------------
    # FIGURE 3: Confusion Matrices
    # ------------------------------------------------------------------
    def generate_fig3_confusion_matrices(self):
        """Figure 3 - Confusion matrices for best-performing models."""
        print("\n" + "="*80)
        print("[Fig 3/6] CONFUSION MATRICES")
        print("="*80)

        cm_individual_dir = self.fig_dirs['fig3'] / "individual"

        # Clean existing
        for f in cm_individual_dir.glob("*.png"):
            f.unlink()
        for f in self.fig_dirs['fig3'].glob("*.png"):
            f.unlink()

        # Step 1: Regenerate publication-quality CMs
        print("  Regenerating publication-quality confusion matrices...")
        try:
            result = subprocess.run([
                sys.executable,
                str(project_root / "scripts" / "visualization" / "regenerate_publication_quality_confusion_matrices.py")
            ], capture_output=True, text=True, timeout=900)

            if result.returncode == 0:
                print("  [OK] Regenerated all confusion matrices")
            else:
                print("  [WARNING] Some confusion matrices failed to regenerate")
                if result.stderr:
                    print(f"  {result.stderr[:200]}")
        except Exception as e:
            print(f"  [WARNING] Could not regenerate CMs: {e}")

        # Step 2: Collect individual CMs from experiment folders
        print("  Collecting individual confusion matrices...")
        count = 0

        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if not exp_dir.is_dir() or not exp_dir.name.startswith('experiment_'):
                continue

            dataset_name = exp_dir.name.replace('experiment_', '')

            for cls_dir in sorted(exp_dir.glob("cls_*_focal")):
                cm_file = cls_dir / "confusion_matrix.png"
                if cm_file.exists():
                    model_name = cls_dir.name.replace('cls_', '').replace('_focal', '')
                    dest_name = f"{dataset_name}_{model_name}.png"
                    shutil.copy2(cm_file, cm_individual_dir / dest_name)
                    count += 1

        print(f"  [OK] Collected {count} individual confusion matrices")
        self.results['fig3'] = count

    # ------------------------------------------------------------------
    # FIGURE 4: Training Curves
    # ------------------------------------------------------------------
    def generate_fig4_training_curves(self):
        """Figure 4 - Training accuracy curves (best vs worst per dataset)."""
        from scripts.visualization.generate_training_curves import TrainingCurvesGenerator

        print("\n" + "="*80)
        print("[Fig 4/6] TRAINING CURVES")
        print("="*80)

        curves_dir = self.fig_dirs['fig4']

        # Clean existing
        for f in curves_dir.glob("*.png"):
            f.unlink()
        metadata_file = curves_dir / "training_curves_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()

        try:
            generator = TrainingCurvesGenerator(
                experiment_dir=self.experiments_dir,
                output_dir=curves_dir,
                dpi=400
            )
            results = generator.generate_all(plot_types=['accuracy', 'loss'])
            total = sum(results.values())
            print(f"  [OK] Generated {total} training curve figures")
            self.results['fig4'] = total
        except Exception as e:
            print(f"  [ERROR] Failed to generate training curves: {e}")
            self.results['fig4'] = 0

    # ------------------------------------------------------------------
    # FIGURES 5 & 6: Detection & Classification Error Cases
    # ------------------------------------------------------------------
    def _load_detection_metadata(self, dataset: str, model: str = 'yolo11') -> 'pd.DataFrame':
        """Load detection metadata CSV for a dataset/model pair."""
        import pandas as pd
        viz_dir = self.experiments_dir / f"experiment_{dataset}" / "visualizations"
        candidates = list(viz_dir.glob(f"pred_detection_{model}*/detection_metadata.csv"))
        if not candidates:
            return pd.DataFrame()
        df = pd.read_csv(candidates[0])
        df['_viz_dir'] = str(candidates[0].parent)
        df['_model'] = candidates[0].parent.name.replace('pred_detection_', '')
        return df

    def _load_classification_metadata(self, dataset: str, model_pattern: str = '*') -> 'pd.DataFrame':
        """Load classification metadata CSV, searching across models."""
        import pandas as pd
        viz_dir = self.experiments_dir / f"experiment_{dataset}" / "visualizations"
        candidates = list(viz_dir.glob(f"pred_classification_{model_pattern}/classification_metadata_images.csv"))
        all_dfs = []
        for csv_path in candidates:
            df = pd.read_csv(csv_path)
            df['_viz_dir'] = str(csv_path.parent)
            df['_model'] = csv_path.parent.name.replace('pred_classification_', '')
            all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def _pick_best_row(self, df: 'pd.DataFrame', condition, sort_cols, ascending, label: str):
        """Filter df by condition, sort, return best row or None."""
        filtered = df[condition].copy()
        if filtered.empty:
            print(f"    [WARNING] No match for {label}")
            return None
        filtered = filtered.sort_values(sort_cols, ascending=ascending)
        return filtered.iloc[0]

    def _copy_case(self, row, dest_dir: Path, fig_label: str) -> bool:
        """Copy a single selected case image to dest_dir."""
        if row is None:
            return False
        viz_dir = Path(row['_viz_dir'])
        src = viz_dir / f"{row['image_name']}.png"
        if not src.exists():
            # Try sanitized name
            src = viz_dir / f"{sanitize_filename(row['image_name'])}.png"
        if src.exists():
            dest = dest_dir / f"{fig_label}_{sanitize_filename(row['image_name'])}.png"
            shutil.copy2(src, dest)
            print(f"    [{fig_label}] {row['image_name']}")
            return True
        else:
            print(f"    [{fig_label}] NOT FOUND: {src}")
            return False

    def generate_fig5_fig6_error_cases(self):
        """
        Figures 5 & 6 - Exactly 6 images each, matching journal paper layout.

        Figure 5 (Detection): YOLOv11 Detection Error Patterns
          (a) IML Lifecycle - False positive case
          (b) IML Lifecycle - False negative case
          (c) MP-IDB Stages - Overdetection (high FP count)
          (d) MP-IDB Species - Mixed errors (FP + FN)
          (e) MD-2019 - Crowded mixed case
          (f) MD-2019 - False negative case

        Figure 6 (Classification): Classification Confusion Patterns
          (a) IML Lifecycle - Single error (1 incorrect, ~3 boxes)
          (b) IML Lifecycle - Moderate error (partial accuracy)
          (c) MP-IDB Stages - Stage confusion (high error count)
          (d) MP-IDB Species - Species misidentification (0% accuracy)
          (e) MD-2019 - Heavy error (low accuracy, many boxes)
          (f) MD-2019 - Perfect classification (100% accuracy, many boxes)
        """
        import pandas as pd
        from scripts.visualization.reporters import CSVReporter, MarkdownReporter

        print("\n" + "="*80)
        print("[Fig 5-6/6] DETECTION & CLASSIFICATION ERROR CASES")
        print("="*80)

        det_dir = self.fig_dirs['fig5']
        cls_dir = self.fig_dirs['fig6']
        metadata_dir = self.output_dir / "metadata"

        # Clean existing
        for d in [det_dir, cls_dir]:
            if d.exists():
                for f in d.glob("*.png"):
                    f.unlink()

        # ================================================================
        # FIGURE 5: Detection errors (6 specific cases matching paper)
        # Paper: "YOLOv11 Detection Error Patterns: (a-b) IML Lifecycle
        #   false positive/negative, (c-d) MP-IDB Stages/Species
        #   overdetection and crowding, (e-f) MD-2019 inter-patient variation"
        # ================================================================
        print("\n  [Fig 5] Selecting 6 detection error cases...")

        det_iml = self._load_detection_metadata('iml_lifecycle', 'yolo11')
        det_md = self._load_detection_metadata('md_2019_stages', 'yolo11')
        det_species = self._load_detection_metadata('mp_idb_species', 'yolo11')

        import pandas as pd
        # Paper caption: "YOLOv11 Detection Error Patterns" — use YOLO11 for all
        det_stages = self._load_detection_metadata('mp_idb_stages', 'yolo11')

        fig5_cases = {}

        # 5a: IML - FP only (most FPs, no FN)
        fig5_cases['5a'] = self._pick_best_row(
            det_iml,
            (det_iml['n_false_positives'] > 0) & (det_iml['n_false_negatives'] == 0),
            ['n_false_positives', 'avg_confidence'], [False, False],
            "5a IML FP"
        )

        # 5b: IML - FN only (most FNs, no FP)
        fig5_cases['5b'] = self._pick_best_row(
            det_iml,
            (det_iml['n_false_negatives'] > 0) & (det_iml['n_false_positives'] == 0),
            ['n_false_negatives'], [False],
            "5b IML FN"
        )

        # 5c: MP-IDB Stages - Paper (updated): "6 false positives and 5 false negatives"
        #   Use YOLO11 (consistent with caption), pick worst overdetection case
        fig5_cases['5c'] = self._pick_best_row(
            det_stages,
            (det_stages['n_false_positives'] > 0),
            ['n_false_positives', 'n_false_negatives'], [False, False],
            "5c Stages YOLO11 worst-overdetection"
        )

        # 5d: MP-IDB Species - Mixed (both FP and FN, most errors)
        fig5_cases['5d'] = self._pick_best_row(
            det_species,
            (det_species['n_false_positives'] > 0) & (det_species['n_false_negatives'] > 0),
            ['n_false_positives', 'n_false_negatives'], [False, False],
            "5d Species mixed"
        )

        # 5e: MD-2019 - Crowded mixed (both FP and FN, many GT boxes)
        if not det_md.empty:
            det_md_mixed = det_md[
                (det_md['n_false_positives'] > 0) & (det_md['n_false_negatives'] > 0)
            ].copy()
            if not det_md_mixed.empty:
                det_md_mixed['total_errors'] = det_md_mixed['n_false_positives'] + det_md_mixed['n_false_negatives']
                fig5_cases['5e'] = det_md_mixed.sort_values(
                    ['total_errors', 'n_gt_boxes'], ascending=[False, False]
                ).iloc[0]
            else:
                fig5_cases['5e'] = None
        else:
            fig5_cases['5e'] = None

        # 5f: MD-2019 - FN only (atypical morphology, high FN count)
        fig5_cases['5f'] = self._pick_best_row(
            det_md,
            (det_md['n_false_negatives'] > 0) & (det_md['n_false_positives'] == 0),
            ['n_false_negatives'], [False],
            "5f MD-2019 FN"
        )

        # Copy fig5 images with stats
        fig5_copied = 0
        fig5_rows = []
        for label in ['5a', '5b', '5c', '5d', '5e', '5f']:
            row = fig5_cases[label]
            if row is not None:
                fp = row.get('n_false_positives', '?')
                fn = row.get('n_false_negatives', '?')
                gt = row.get('n_gt_boxes', '?')
                model = row.get('_model', '?')
                print(f"    [{label}] {row['image_name']} (GT={gt}, FP={fp}, FN={fn}, model={model})")
            if self._copy_case(row, det_dir, label):
                fig5_copied += 1
                fig5_rows.append(row)
        print(f"  [OK] Figure 5: {fig5_copied}/6 images copied")
        self.results['fig5'] = fig5_copied

        # ================================================================
        # FIGURE 6: Classification errors (6 specific cases matching paper)
        # Paper: "Classification Confusion Patterns Using Best Models:
        #   (a-b) IML Lifecycle single/moderate errors,
        #   (c-d) MP-IDB stage/species confusion,
        #   (e-f) MD-2019 heavy errors and perfect classification"
        # ================================================================
        print("\n  [Fig 6] Selecting 6 classification error cases...")

        cls_iml = self._load_classification_metadata('iml_lifecycle')
        cls_md = self._load_classification_metadata('md_2019_stages')
        cls_species = self._load_classification_metadata('mp_idb_species')
        cls_stages = self._load_classification_metadata('mp_idb_stages')

        fig6_cases = {}

        # 6a: IML - Paper: "1 error among 3 parasites" using EfficientNet-B1
        cls_iml_b1 = cls_iml[cls_iml['_model'] == 'efficientnet_b1_focal']
        if cls_iml_b1.empty:
            cls_iml_b1 = cls_iml  # fallback to any model
        fig6_cases['6a'] = self._pick_best_row(
            cls_iml_b1,
            (cls_iml_b1['n_boxes'] == 3) & (cls_iml_b1['n_incorrect'] == 1),
            ['avg_confidence'], [False],
            "6a IML B1 3-box 1-error"
        )

        # 6b: IML - Paper: "1 error among 3 parasites, 66.7%" using EfficientNet-B1
        used_6a = fig6_cases['6a']['image_name'] if fig6_cases['6a'] is not None else ''
        fig6_cases['6b'] = self._pick_best_row(
            cls_iml_b1,
            (cls_iml_b1['n_boxes'] == 3) & (cls_iml_b1['n_incorrect'] == 1) &
            (cls_iml_b1['image_name'] != used_6a),
            ['avg_confidence'], [True],
            "6b IML B1 3-box 1-error (diff)"
        )

        # 6c: MP-IDB Stages - Paper (updated): "1 misclassification among 31
        #   parasites at 96.8% accuracy with ResNet101"
        cls_stages_r101 = cls_stages[cls_stages['_model'] == 'resnet101_focal']
        if cls_stages_r101.empty:
            cls_stages_r101 = cls_stages
        row_6c = self._pick_best_row(
            cls_stages_r101,
            (cls_stages_r101['n_boxes'] >= 10) & (cls_stages_r101['n_incorrect'] > 0),
            ['accuracy', 'n_boxes'], [False, False],
            "6c Stages ResNet101 best-acc high-box"
        )
        fig6_cases['6c'] = row_6c

        # 6d: MP-IDB Species - Paper: "100% misclassification, single parasite"
        fig6_cases['6d'] = self._pick_best_row(
            cls_species,
            (cls_species['accuracy'] == 0) & (cls_species['n_boxes'] <= 3),
            ['avg_confidence'], [False],
            "6d Species 0%-acc"
        )

        # 6e: MD-2019 - Paper: "6 errors among 8, 25% accuracy"
        row_6e = self._pick_best_row(
            cls_md,
            (cls_md['n_boxes'] == 8) & (cls_md['n_incorrect'] == 6),
            ['avg_confidence'], [False],
            "6e MD-2019 8-box 6-error exact"
        )
        if row_6e is None:
            # Fallback: closest to 25% accuracy with ~8 boxes
            row_6e = self._pick_best_row(
                cls_md,
                (cls_md['n_boxes'] >= 6) & (cls_md['n_boxes'] <= 10) &
                (cls_md['accuracy'] >= 0.2) & (cls_md['accuracy'] <= 0.3),
                ['n_boxes'], [False],
                "6e MD-2019 ~8-box ~25% relaxed"
            )
        fig6_cases['6e'] = row_6e

        # 6f: MD-2019 - Paper: "10 parasites, 100% accuracy"
        #   Try n_boxes=10 first, then relax to >=8
        row_6f = self._pick_best_row(
            cls_md,
            (cls_md['n_boxes'] == 10) & (cls_md['accuracy'] == 1.0),
            ['avg_confidence'], [False],
            "6f MD-2019 10-box perfect exact"
        )
        if row_6f is None:
            row_6f = self._pick_best_row(
                cls_md,
                (cls_md['accuracy'] == 1.0) & (cls_md['n_boxes'] >= 8),
                ['n_boxes', 'avg_confidence'], [False, False],
                "6f MD-2019 >=8-box perfect"
            )
        fig6_cases['6f'] = row_6f

        # Copy fig6 images with stats
        fig6_copied = 0
        fig6_rows = []
        for label in ['6a', '6b', '6c', '6d', '6e', '6f']:
            row = fig6_cases[label]
            if row is not None:
                boxes = row.get('n_boxes', '?')
                correct = row.get('n_correct', '?')
                wrong = row.get('n_incorrect', '?')
                acc = row.get('accuracy', '?')
                model = row.get('_model', '?')
                print(f"    [{label}] {row['image_name']} (boxes={boxes}, correct={correct}, wrong={wrong}, acc={acc}, model={model})")
            if self._copy_case(row, cls_dir, label):
                fig6_copied += 1
                fig6_rows.append(row)
        print(f"  [OK] Figure 6: {fig6_copied}/6 images copied")
        self.results['fig6'] = fig6_copied

        # ================================================================
        # Save metadata
        # ================================================================
        detection_df = pd.DataFrame([r for r in fig5_rows if r is not None])
        classification_df = pd.DataFrame([r for r in fig6_rows if r is not None])

        csv_reporter = CSVReporter()
        csv_reporter.generate(detection_df, metadata_dir / "selected_detection_errors.csv")
        csv_reporter.generate(classification_df, metadata_dir / "selected_classification_errors.csv")

        combined = pd.concat([detection_df, classification_df], ignore_index=True)
        csv_reporter.generate(combined, metadata_dir / "selected_error_images.csv")

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
        md_reporter.generate(summary_data, metadata_dir / "visualization_report.md")

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    def create_readme(self):
        """Create README guide with figure mapping."""
        readme_path = self.output_dir / "_README.txt"
        content = f"""CENTRALIZED VISUALIZATION OUTPUTS
==================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Experiment: {self.experiment_root}

PAPER FIGURE MAPPING
--------------------
Figure 1 (Augmentation)     -> {FIGURE_DIRS['fig1']}/
Figure 2 (Pipeline)         -> {FIGURE_DIRS['fig2']}/
Figure 3 (Confusion Matrix) -> {FIGURE_DIRS['fig3']}/individual/
Figure 4 (Training Curves)  -> {FIGURE_DIRS['fig4']}/
Figure 5 (Detection Errors) -> {FIGURE_DIRS['fig5']}/
Figure 6 (Classification)   -> {FIGURE_DIRS['fig6']}/

GENERATION RESULTS
------------------
Figure 1: {self.results.get('fig1', '-')} file(s)
Figure 2: {self.results.get('fig2', '-')} file(s)
Figure 3: {self.results.get('fig3', '-')} file(s)
Figure 4: {self.results.get('fig4', '-')} file(s)
Figure 5: {self.results.get('fig5', '-')} file(s)
Figure 6: {self.results.get('fig6', '-')} file(s)

METADATA
--------
metadata/selected_detection_errors.csv
metadata/selected_classification_errors.csv
metadata/selected_error_images.csv
metadata/visualization_report.md

TO REGENERATE
-------------
python scripts/visualization/generate_all_centralized.py
python scripts/visualization/generate_all_centralized.py --only fig1 fig3
"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [OK] Created: _README.txt")

    # ------------------------------------------------------------------
    # MAIN RUN
    # ------------------------------------------------------------------
    def run(self, only_figures: List[str] = None):
        """
        Run figure generation.

        Args:
            only_figures: List of figure keys to generate (e.g. ['fig1', 'fig3']).
                         None = generate all.
        """
        all_figures = ['fig1', 'fig2', 'fig3', 'fig4', 'fig5_fig6']
        generate_all = only_figures is None

        if generate_all:
            targets = all_figures
        else:
            # Normalize: if user says fig5 or fig6, include fig5_fig6
            targets = []
            for f in only_figures:
                if f in ('fig5', 'fig6'):
                    if 'fig5_fig6' not in targets:
                        targets.append('fig5_fig6')
                elif f in ('fig1', 'fig2', 'fig3', 'fig4'):
                    targets.append(f)

        print("\n" + "="*80)
        print("PAPER FIGURE GENERATION")
        print("="*80)
        print(f"Experiment: {self.experiment_root}")
        print(f"Output:     {self.output_dir.absolute()}")
        print(f"Figures:    {', '.join(targets) if not generate_all else 'ALL (fig1-fig6)'}")
        print("="*80)

        # Execute each figure generator
        generators = {
            'fig1':     self.generate_fig1_augmentation,
            'fig2':     self.generate_fig2_pipeline,
            'fig3':     self.generate_fig3_confusion_matrices,
            'fig4':     self.generate_fig4_training_curves,
            'fig5_fig6': self.generate_fig5_fig6_error_cases,
        }

        skip_labels = {
            'fig1': FIGURE_LABELS['fig1'],
            'fig2': FIGURE_LABELS['fig2'],
            'fig3': FIGURE_LABELS['fig3'],
            'fig4': FIGURE_LABELS['fig4'],
            'fig5_fig6': 'Figures 5 & 6 - Detection & Classification Errors',
        }

        for key in all_figures:
            if key in targets:
                generators[key]()
            else:
                print(f"\n[SKIPPED] {skip_labels[key]}")

        self.create_readme()

        # Summary
        print("\n" + "="*80)
        print("GENERATION COMPLETE")
        print("="*80)
        for fig_key in ['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6']:
            count = self.results.get(fig_key, '-')
            label = FIGURE_LABELS[fig_key]
            status = f"{count} file(s)" if isinstance(count, int) and count > 0 else "skipped"
            print(f"  {label}: {status}")
        print(f"\nOutput: {self.output_dir.absolute()}")
        print("="*80)


def auto_detect_experiment() -> str:
    """Find the latest results/optA_* experiment folder."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None

    experiments = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('optA_')],
        key=lambda x: x.name, reverse=True
    )
    return str(experiments[0]) if experiments else None


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(
        description='Generate ALL paper figures to one centralized folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Paper Figure Mapping:
  fig1  = Figure 1 (Augmentation examples)
  fig2  = Figure 2 (Pipeline architecture)
  fig3  = Figure 3 (Confusion matrices)
  fig4  = Figure 4 (Training curves)
  fig5  = Figure 5 (Detection errors)
  fig6  = Figure 6 (Classification errors)

Examples:
  # Generate ALL figures
  python scripts/visualization/generate_all_centralized.py

  # Generate specific figures only
  python scripts/visualization/generate_all_centralized.py --only fig1 fig3 fig4

  # Generate only error case figures
  python scripts/visualization/generate_all_centralized.py --only fig5 fig6

  # Custom experiment directory
  python scripts/visualization/generate_all_centralized.py --experiment results/optA_20251016_200330
        """
    )
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment directory (default: auto-detect latest)')
    parser.add_argument('--output', type=str, default='visualization_outputs',
                       help='Output directory (default: visualization_outputs/)')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of error cases per category (default: 20)')
    parser.add_argument('--only', nargs='+', choices=['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6'],
                       help='Generate only specified figures (e.g. --only fig1 fig3)')

    args = parser.parse_args()

    # Auto-detect experiment
    if args.experiment is None:
        args.experiment = auto_detect_experiment()
        if args.experiment is None:
            print("[ERROR] No experiments found in results/")
            return 1
        print(f"[AUTO-DETECT] Using: {args.experiment}")

    generator = CentralizedVisualizationGenerator(
        experiment_root=Path(args.experiment),
        output_dir=Path(args.output),
        top_n=args.top_n
    )
    generator.run(only_figures=args.only)
    return 0


if __name__ == '__main__':
    sys.exit(main())
