#!/usr/bin/env python3
"""
Verify Publication Data Integrity

This script verifies that all claims made in publication documents match actual experimental results.
Prevents fabricated data, inflated metrics, and incorrect best model claims.

Usage:
    python scripts/publication/verify_publication_data.py --experiment results/optA_LATEST
    python scripts/publication/verify_publication_data.py --experiment results/optA_20251012_143000 --paper luaran/JICEST_Paper.md
    python scripts/publication/verify_publication_data.py --experiment results/optA_20251012_143000 --strict
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import numpy as np

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class PublicationVerifier:
    """Verify publication data integrity against actual experimental results."""

    def __init__(self, experiment_dir: str, paper_path: str = None, strict: bool = False):
        """
        Initialize verifier with experiment directory and optional paper path.

        Args:
            experiment_dir: Path to experiment results directory
            paper_path: Optional path to paper markdown file to verify
            strict: If True, warnings become failures
        """
        self.exp_dir = Path(experiment_dir)
        self.paper_path = Path(paper_path) if paper_path else None
        self.strict = strict

        # Verification results storage
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'checks_performed': 0
        }

        # Ground truth data storage
        self.ground_truth = {}
        self.paper_claims = {}

        # Expected components
        self.expected_datasets = ['iml_lifecycle', 'mp_idb_species', 'mp_idb_stages']
        self.expected_detection_models = ['yolo10', 'yolo11', 'yolo12']
        self.expected_classification_models = [
            'densenet121', 'efficientnet_b0', 'efficientnet_b1',
            'efficientnet_b2', 'resnet50', 'resnet101'
        ]

        # Critical minority classes to check
        self.minority_classes = {
            'mp_idb_species': ['P_ovale', 'P_malariae'],
            'mp_idb_stages': ['schizont', 'gametocyte'],
            'iml_lifecycle': ['schizont', 'gametocyte']
        }

    def load_ground_truth(self) -> bool:
        """Load actual results from experiment directory."""
        try:
            # Check if experiment directory exists
            if not self.exp_dir.exists():
                self.results['failed'].append(f"❌ Experiment directory not found: {self.exp_dir}")
                return False

            # Load consolidated analysis if available (multi-dataset)
            consolidated_path = self.exp_dir / 'consolidated_analysis' / 'cross_dataset_comparison'
            if consolidated_path.exists():
                self.ground_truth['multi_dataset'] = True
                self._load_consolidated_results(consolidated_path)

            # Load individual dataset results
            experiments_dir = self.exp_dir / 'experiments'
            if experiments_dir.exists():
                for dataset_dir in experiments_dir.glob('experiment_*'):
                    dataset_name = dataset_dir.name.replace('experiment_', '')
                    self.ground_truth[dataset_name] = self._load_dataset_results(dataset_dir)

            # Load master summary if available
            master_summary = self.exp_dir / 'master_summary.json'
            if master_summary.exists():
                with open(master_summary, 'r') as f:
                    self.ground_truth['master_summary'] = json.load(f)

            if not self.ground_truth:
                self.results['failed'].append("❌ No results found in experiment directory")
                return False

            self.results['passed'].append(f"✅ Loaded ground truth from {self.exp_dir.name}")
            return True

        except Exception as e:
            self.results['failed'].append(f"❌ Error loading ground truth: {str(e)}")
            return False

    def _load_consolidated_results(self, consolidated_path: Path):
        """Load consolidated cross-dataset comparison results."""
        # Load detection performance
        det_csv = consolidated_path / 'detection_performance_all_datasets.csv'
        if det_csv.exists():
            self.ground_truth['detection_consolidated'] = pd.read_csv(det_csv)

        # Load classification performance (both loss types)
        focal_csv = consolidated_path / 'classification_focal_loss_all_datasets.csv'
        if focal_csv.exists():
            self.ground_truth['classification_focal'] = pd.read_csv(focal_csv)

        cb_csv = consolidated_path / 'classification_class_balanced_all_datasets.csv'
        if cb_csv.exists():
            self.ground_truth['classification_cb'] = pd.read_csv(cb_csv)

        # Load comprehensive summary JSON
        summary_json = consolidated_path / 'comprehensive_summary.json'
        if summary_json.exists():
            with open(summary_json, 'r') as f:
                self.ground_truth['comprehensive_summary'] = json.load(f)

    def _load_dataset_results(self, dataset_dir: Path) -> Dict:
        """Load results for a specific dataset."""
        results = {}

        # Load Table 9 classification pivot
        table9_focal = dataset_dir / 'table9_focal_loss.csv'
        if table9_focal.exists():
            results['table9_focal'] = pd.read_csv(table9_focal)

        table9_cb = dataset_dir / 'table9_class_balanced.csv'
        if table9_cb.exists():
            results['table9_cb'] = pd.read_csv(table9_cb)

        # Load detection results
        for model in self.expected_detection_models:
            det_dir = dataset_dir / f'det_{model}'
            if det_dir.exists():
                results_csv = det_dir / 'results.csv'
                if results_csv.exists():
                    if f'detection_{model}' not in results:
                        results[f'detection_{model}'] = {}
                    results[f'detection_{model}']['results'] = pd.read_csv(results_csv)

        # Load classification analysis results
        for analysis_dir in dataset_dir.glob('analysis_classification_*'):
            model_name = analysis_dir.name.replace('analysis_classification_', '')
            metrics_file = analysis_dir / 'test_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    if 'classification_analysis' not in results:
                        results['classification_analysis'] = {}
                    results['classification_analysis'][model_name] = json.load(f)

        return results

    def extract_paper_claims(self) -> bool:
        """Extract claims from paper markdown if provided."""
        if not self.paper_path:
            return True  # Skip if no paper provided

        try:
            if not self.paper_path.exists():
                self.results['warnings'].append(f"⚠️ Paper file not found: {self.paper_path}")
                return True  # Continue with other checks

            with open(self.paper_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract performance numbers using regex
            self.paper_claims = self._extract_metrics_from_text(content)

            if self.paper_claims:
                self.results['passed'].append(f"✅ Extracted claims from {self.paper_path.name}")
            else:
                self.results['warnings'].append(f"⚠️ No metrics found in paper")

            return True

        except Exception as e:
            self.results['warnings'].append(f"⚠️ Error reading paper: {str(e)}")
            return True  # Continue with other checks

    def _extract_metrics_from_text(self, text: str) -> Dict:
        """Extract metric values from paper text."""
        claims = {}

        # Common patterns for metrics
        patterns = {
            'accuracy': r'accuracy[:\s]+([0-9.]+)%?',
            'precision': r'precision[:\s]+([0-9.]+)%?',
            'recall': r'recall[:\s]+([0-9.]+)%?',
            'f1_score': r'F1[\-\s]?score[:\s]+([0-9.]+)%?',
            'map50': r'mAP@0?\.?50?[:\s]+([0-9.]+)%?',
            'best_model': r'best[\s\w]*model[:\s]+(\w+)',
        }

        for metric, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                claims[metric] = matches

        # Extract table data if present
        table_pattern = r'\|[^\n]+\|[^\n]+\|'
        tables = re.findall(table_pattern, text)
        if tables:
            claims['tables'] = tables

        return claims

    def verify_best_model_claims(self):
        """Verify that claimed best models actually have the highest metrics."""
        self.results['checks_performed'] += 1
        issues = []
        verified = []

        # Check each dataset
        for dataset in self.expected_datasets:
            if dataset not in self.ground_truth:
                continue

            dataset_results = self.ground_truth[dataset]

            # Verify best detection model
            best_det_model, best_det_score = self._find_best_detection_model(dataset_results)
            if best_det_model:
                verified.append(f"  ✅ {dataset} detection: {best_det_model} (mAP@0.5: {best_det_score:.4f})")

            # Verify best classification model (Focal Loss)
            if 'table9_focal' in dataset_results:
                best_cls_model, best_cls_score = self._find_best_classification_model(
                    dataset_results['table9_focal']
                )
                if best_cls_model:
                    verified.append(f"  ✅ {dataset} classification: {best_cls_model} (Acc: {best_cls_score:.4f})")

                    # Check if paper claims match
                    if self.paper_claims and 'best_model' in self.paper_claims:
                        claimed_models = self.paper_claims['best_model']
                        if best_cls_model not in str(claimed_models):
                            issues.append(f"  ❌ {dataset}: Paper claims different best model!")

        # Report results
        if issues:
            self.results['failed'].append("❌ Best Model Claims FAILED")
            self.results['failed'].extend(issues)
        else:
            self.results['passed'].append("✅ Best Model Claims PASSED")
            self.results['passed'].extend(verified)

    def _find_best_detection_model(self, dataset_results: Dict) -> Tuple[Optional[str], Optional[float]]:
        """Find the best detection model for a dataset."""
        best_model = None
        best_score = 0.0

        for model in self.expected_detection_models:
            key = f'detection_{model}'
            if key in dataset_results and 'results' in dataset_results[key]:
                df = dataset_results[key]['results']
                # Get the last row (best epoch) mAP@0.5
                if 'metrics/mAP50(B)' in df.columns:
                    score = df['metrics/mAP50(B)'].iloc[-1]
                    if score > best_score:
                        best_score = score
                        best_model = model

        return best_model, best_score

    def _find_best_classification_model(self, table9_df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """Find the best classification model from Table 9."""
        # Table 9 format: Model | Accuracy | Balanced_Accuracy | ...
        if 'Accuracy' not in table9_df.columns:
            return None, None

        best_idx = table9_df['Accuracy'].idxmax()
        best_model = table9_df.loc[best_idx, 'Model']
        best_score = table9_df.loc[best_idx, 'Accuracy']

        return best_model, best_score

    def verify_performance_numbers(self):
        """Verify that performance numbers aren't fabricated or inflated."""
        self.results['checks_performed'] += 1
        issues = []
        verified = []

        # Define acceptable tolerance for metric differences
        tolerance = 0.01  # 1% tolerance

        if not self.paper_claims:
            self.results['warnings'].append("⚠️ No paper claims to verify performance against")
            return

        # Check claimed accuracies against actual
        if 'accuracy' in self.paper_claims:
            for claimed_acc in self.paper_claims['accuracy']:
                try:
                    claimed_val = float(claimed_acc.replace('%', '')) / 100 if '%' in claimed_acc else float(claimed_acc)

                    # Find closest actual accuracy
                    closest_match = self._find_closest_metric_match('accuracy', claimed_val)
                    if closest_match:
                        actual_val, source = closest_match
                        diff = abs(claimed_val - actual_val)

                        if diff > tolerance:
                            issues.append(f"  ❌ Accuracy claim {claimed_val:.4f} differs from actual {actual_val:.4f} (source: {source})")
                        else:
                            verified.append(f"  ✅ Accuracy {claimed_val:.4f} verified (actual: {actual_val:.4f})")
                except ValueError:
                    pass

        # Report results
        if issues:
            self.results['failed'].append("❌ Performance Numbers FAILED")
            self.results['failed'].extend(issues)
        else:
            self.results['passed'].append("✅ Performance Numbers PASSED")
            self.results['passed'].extend(verified)

    def _find_closest_metric_match(self, metric_type: str, value: float) -> Optional[Tuple[float, str]]:
        """Find the closest matching metric value in ground truth."""
        closest = None
        min_diff = float('inf')

        # Search through all datasets
        for dataset_name, dataset_data in self.ground_truth.items():
            if isinstance(dataset_data, dict):
                # Check classification analysis
                if 'classification_analysis' in dataset_data:
                    for model_name, metrics in dataset_data['classification_analysis'].items():
                        if metric_type in metrics:
                            actual_val = metrics[metric_type]
                            diff = abs(value - actual_val)
                            if diff < min_diff:
                                min_diff = diff
                                closest = (actual_val, f"{dataset_name}/{model_name}")

        return closest if min_diff < 0.1 else None  # Only return if reasonably close

    def verify_per_class_metrics(self):
        """Verify per-class metrics, especially for minority classes."""
        self.results['checks_performed'] += 1
        issues = []
        verified = []
        critical_issues = []

        for dataset_name, minority_classes in self.minority_classes.items():
            if dataset_name not in self.ground_truth:
                continue

            dataset_results = self.ground_truth[dataset_name]

            # Check classification analysis for per-class metrics
            if 'classification_analysis' in dataset_results:
                for model_name, metrics in dataset_results['classification_analysis'].items():
                    if 'per_class_metrics' in metrics:
                        per_class = metrics['per_class_metrics']

                        for class_name in minority_classes:
                            if class_name in per_class:
                                class_metrics = per_class[class_name]

                                # Check for suspiciously high minority class performance
                                if 'recall' in class_metrics:
                                    recall = class_metrics['recall']

                                    # Flag if minority class has perfect or near-perfect recall
                                    if recall >= 0.95:
                                        critical_issues.append(
                                            f"  🚨 SUSPICIOUS: {dataset_name}/{model_name} - "
                                            f"{class_name} recall = {recall:.2%} (minority class with perfect recall?)"
                                        )
                                    elif recall < 0.3:
                                        verified.append(
                                            f"  ✅ {dataset_name}/{model_name} - "
                                            f"{class_name} recall = {recall:.2%} (realistic for minority class)"
                                        )

                                # Check F1 score
                                if 'f1_score' in class_metrics:
                                    f1 = class_metrics['f1_score']
                                    if f1 < 0.5:
                                        verified.append(
                                            f"  ✅ {dataset_name} - {class_name} F1 = {f1:.4f} (expected for minority)"
                                        )

        # Special check for P. ovale (known fabrication case)
        p_ovale_check = self._verify_p_ovale_specifically()
        if p_ovale_check:
            critical_issues.append(p_ovale_check)

        # Report results
        if critical_issues:
            self.results['failed'].append("❌ Per-Class Metrics FAILED (CRITICAL)")
            self.results['failed'].extend(critical_issues)

        if issues:
            self.results['failed'].extend(issues)

        if verified:
            self.results['passed'].append("✅ Per-Class Metrics Realistic")
            self.results['passed'].extend(verified)

    def _verify_p_ovale_specifically(self) -> Optional[str]:
        """Special verification for P. ovale metrics (known fabrication case)."""
        if 'mp_idb_species' in self.ground_truth:
            dataset = self.ground_truth['mp_idb_species']
            if 'classification_analysis' in dataset:
                for model_name, metrics in dataset['classification_analysis'].items():
                    if 'per_class_metrics' in metrics and 'P_ovale' in metrics['per_class_metrics']:
                        p_ovale_metrics = metrics['per_class_metrics']['P_ovale']
                        if 'recall' in p_ovale_metrics:
                            recall = p_ovale_metrics['recall']
                            if recall > 0.9:
                                return (f"  🚨 CRITICAL: P. ovale recall = {recall:.2%} in {model_name} "
                                       f"(Historical fabrication detected! Expected ~20%)")
        return None

    def verify_consistency(self):
        """Verify consistency across different outputs."""
        self.results['checks_performed'] += 1
        issues = []
        verified = []

        # Check Table 9 consistency with classification analysis
        for dataset_name in self.expected_datasets:
            if dataset_name not in self.ground_truth:
                continue

            dataset_results = self.ground_truth[dataset_name]

            # Compare Table 9 with classification analysis
            if 'table9_focal' in dataset_results and 'classification_analysis' in dataset_results:
                table9 = dataset_results['table9_focal']

                for _, row in table9.iterrows():
                    model_name = row['Model']
                    table9_acc = row['Accuracy']

                    # Find corresponding classification analysis
                    analysis_key = None
                    for key in dataset_results['classification_analysis']:
                        if model_name.lower() in key.lower():
                            analysis_key = key
                            break

                    if analysis_key:
                        analysis_acc = dataset_results['classification_analysis'][analysis_key].get('accuracy', 0)
                        diff = abs(table9_acc - analysis_acc)

                        if diff > 0.01:  # 1% tolerance
                            issues.append(
                                f"  ❌ {dataset_name} - {model_name}: "
                                f"Table9={table9_acc:.4f} vs Analysis={analysis_acc:.4f}"
                            )
                        else:
                            verified.append(f"  ✅ {dataset_name} - {model_name}: Consistent accuracy")

        # Check consolidated vs individual results consistency
        if 'comprehensive_summary' in self.ground_truth:
            summary = self.ground_truth['comprehensive_summary']

            # Verify dataset counts
            if 'datasets' in summary:
                actual_datasets = len([d for d in self.expected_datasets if d in self.ground_truth])
                claimed_datasets = len(summary['datasets'])

                if actual_datasets == claimed_datasets:
                    verified.append(f"  ✅ Dataset count consistent: {actual_datasets}")
                else:
                    issues.append(f"  ❌ Dataset count mismatch: claimed={claimed_datasets}, actual={actual_datasets}")

        # Report results
        if issues:
            self.results['failed'].append("❌ Consistency Checks FAILED")
            self.results['failed'].extend(issues)
        else:
            self.results['passed'].append("✅ Consistency Checks PASSED")
            self.results['passed'].extend(verified)

    def verify_completeness(self):
        """Verify all expected outputs are present."""
        self.results['checks_performed'] += 1
        missing = []
        present = []

        # Expected files for each dataset
        expected_files = {
            'table9_focal_loss.csv': 'Table 9 Focal Loss',
            'table9_class_balanced.csv': 'Table 9 Class-Balanced',
            'table9_classification_pivot.xlsx': 'Table 9 Excel Pivot'
        }

        # Check each dataset
        for dataset in self.expected_datasets:
            dataset_dir = self.exp_dir / 'experiments' / f'experiment_{dataset}'

            if not dataset_dir.exists():
                missing.append(f"  ⚠️ Missing dataset directory: {dataset}")
                continue

            # Check expected files
            for filename, description in expected_files.items():
                file_path = dataset_dir / filename
                if file_path.exists():
                    present.append(f"  ✅ {dataset}: {description}")
                else:
                    missing.append(f"  ⚠️ {dataset}: Missing {description}")

            # Check detection models
            for model in self.expected_detection_models:
                det_dir = dataset_dir / f'det_{model}'
                if not det_dir.exists():
                    missing.append(f"  ⚠️ {dataset}: Missing detection model {model}")

            # Check classification models (at least some should exist)
            cls_dirs = list(dataset_dir.glob('cls_*'))
            if len(cls_dirs) < 6:  # Expecting at least 6 classification models
                missing.append(f"  ⚠️ {dataset}: Only {len(cls_dirs)} classification models (expected ≥6)")

        # Check consolidated analysis (if multi-dataset)
        if len([d for d in self.expected_datasets if d in self.ground_truth]) > 1:
            consolidated_dir = self.exp_dir / 'consolidated_analysis' / 'cross_dataset_comparison'
            if consolidated_dir.exists():
                present.append("  ✅ Consolidated analysis present")
            else:
                missing.append("  ⚠️ Missing consolidated analysis for multi-dataset experiment")

        # Report results
        if missing:
            if self.strict:
                self.results['failed'].append("❌ Completeness Check FAILED")
                self.results['failed'].extend(missing)
            else:
                self.results['warnings'].append("⚠️ Completeness Check has warnings")
                self.results['warnings'].extend(missing)

        if present:
            self.results['passed'].append("✅ Expected Outputs Found")
            # Only show first 5 to avoid clutter
            self.results['passed'].extend(present[:5])
            if len(present) > 5:
                self.results['passed'].append(f"  ... and {len(present)-5} more")

    def verify_statistical_validity(self):
        """Verify statistical validity of reported results."""
        self.results['checks_performed'] += 1
        issues = []
        verified = []

        # Check for impossible metric combinations
        for dataset_name, dataset_data in self.ground_truth.items():
            if not isinstance(dataset_data, dict):
                continue

            if 'classification_analysis' in dataset_data:
                for model_name, metrics in dataset_data['classification_analysis'].items():
                    # Check if precision > recall significantly (might indicate overfitting)
                    if 'precision' in metrics and 'recall' in metrics:
                        precision = metrics['precision']
                        recall = metrics['recall']

                        if precision > 0.95 and recall < 0.5:
                            issues.append(
                                f"  ⚠️ {dataset_name}/{model_name}: "
                                f"Suspicious - Precision={precision:.2%} but Recall={recall:.2%}"
                            )

                    # Check if F1 score makes sense given precision and recall
                    if all(k in metrics for k in ['precision', 'recall', 'f1_score']):
                        precision = metrics['precision']
                        recall = metrics['recall']
                        f1 = metrics['f1_score']

                        # Calculate expected F1
                        if precision + recall > 0:
                            expected_f1 = 2 * (precision * recall) / (precision + recall)
                            f1_diff = abs(f1 - expected_f1)

                            if f1_diff > 0.02:  # 2% tolerance
                                issues.append(
                                    f"  ❌ {dataset_name}/{model_name}: "
                                    f"F1={f1:.4f} doesn't match P={precision:.4f}, R={recall:.4f} "
                                    f"(expected {expected_f1:.4f})"
                                )
                            else:
                                verified.append(f"  ✅ {dataset_name}/{model_name}: F1 score valid")

        # Report results
        if issues:
            self.results['failed'].append("❌ Statistical Validity QUESTIONABLE")
            self.results['failed'].extend(issues)
        else:
            self.results['passed'].append("✅ Statistical Validity PASSED")
            self.results['passed'].extend(verified[:3])  # Show only first 3

    def generate_report(self) -> str:
        """Generate comprehensive verification report."""
        report_lines = [
            "=" * 60,
            "PUBLICATION DATA VERIFICATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Experiment: {self.exp_dir}",
        ]

        if self.paper_path:
            report_lines.append(f"Paper: {self.paper_path}")

        report_lines.extend(["", "=" * 60, ""])

        # Add verification results
        total_passed = len(self.results['passed'])
        total_failed = len(self.results['failed'])
        total_warnings = len(self.results['warnings'])

        # Passed checks
        if self.results['passed']:
            report_lines.append("[PASSED CHECKS]")
            report_lines.extend(self.results['passed'])
            report_lines.append("")

        # Failed checks
        if self.results['failed']:
            report_lines.append("[FAILED CHECKS]")
            report_lines.extend(self.results['failed'])
            report_lines.append("")

        # Warnings
        if self.results['warnings']:
            report_lines.append("[WARNINGS]")
            report_lines.extend(self.results['warnings'])
            report_lines.append("")

        # Summary
        report_lines.extend([
            "=" * 60,
            "SUMMARY",
            "=" * 60,
            f"Total Checks Performed: {self.results['checks_performed']}",
            f"Passed Items: {total_passed}",
            f"Failed Items: {total_failed}",
            f"Warnings: {total_warnings}",
            ""
        ])

        # Overall status
        if total_failed > 0:
            report_lines.extend([
                "🚨 CRITICAL: DO NOT SUBMIT - Failed checks detected!",
                "RECOMMENDATION: Fix all failed items before publication submission.",
                ""
            ])

            # Specific recommendations
            if any("P_ovale" in line or "P. ovale" in line for line in self.results['failed']):
                report_lines.append("⚠️ CRITICAL FIX REQUIRED: P. ovale metrics appear fabricated!")

            if any("best model" in line.lower() for line in self.results['failed']):
                report_lines.append("⚠️ FIX REQUIRED: Update best model claims to match actual results!")

        elif total_warnings > 0:
            report_lines.extend([
                "⚠️ STATUS: Review Recommended",
                "RECOMMENDATION: Address warnings for complete publication quality.",
                ""
            ])
        else:
            report_lines.extend([
                "✅ STATUS: ALL CHECKS PASSED",
                "Paper is ready for submission (from data integrity perspective).",
                ""
            ])

        # Add actionable items if there are failures
        if total_failed > 0:
            report_lines.extend([
                "=" * 60,
                "ACTION ITEMS",
                "=" * 60
            ])

            action_items = self._generate_action_items()
            for i, item in enumerate(action_items, 1):
                report_lines.append(f"{i}. {item}")

        return "\n".join(report_lines)

    def _generate_action_items(self) -> List[str]:
        """Generate specific action items based on failures."""
        actions = []

        for failed_item in self.results['failed']:
            if "P_ovale" in failed_item or "P. ovale" in failed_item:
                actions.append("Check and correct P. ovale recall metrics (should be ~20%, not 100%)")
            elif "best model" in failed_item.lower():
                actions.append("Update best model claims in abstract and results sections")
            elif "F1" in failed_item:
                actions.append("Recalculate F1 scores to match precision and recall values")
            elif "accuracy claim" in failed_item.lower():
                actions.append("Verify and correct all accuracy values against actual results")

        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)

        return unique_actions if unique_actions else ["Review all failed checks and correct data accordingly"]

    def run_all_verifications(self) -> bool:
        """Run all verification checks."""
        print("Loading ground truth data...")
        if not self.load_ground_truth():
            return False

        print("Extracting paper claims...")
        self.extract_paper_claims()

        print("\nRunning verification checks...")
        print("1. Verifying best model claims...")
        self.verify_best_model_claims()

        print("2. Verifying performance numbers...")
        self.verify_performance_numbers()

        print("3. Verifying per-class metrics...")
        self.verify_per_class_metrics()

        print("4. Verifying consistency...")
        self.verify_consistency()

        print("5. Verifying completeness...")
        self.verify_completeness()

        print("6. Verifying statistical validity...")
        self.verify_statistical_validity()

        # Return True if no critical failures
        return len(self.results['failed']) == 0


def find_latest_experiment(results_dir: Path) -> Optional[Path]:
    """Find the latest experiment directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return None

    # Look for optA_* directories
    opt_dirs = sorted(results_dir.glob('optA_*'), key=lambda x: x.stat().st_mtime, reverse=True)

    if opt_dirs:
        return opt_dirs[0]

    # Also check for exp_optA_* directories
    exp_dirs = sorted(results_dir.glob('exp_optA_*'), key=lambda x: x.stat().st_mtime, reverse=True)

    return exp_dirs[0] if exp_dirs else None


def main():
    """Main entry point for the verification script."""
    parser = argparse.ArgumentParser(
        description='Verify publication data integrity against experimental results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify latest experiment
  python scripts/publication/verify_publication_data.py --experiment results/optA_LATEST

  # Verify specific experiment with paper
  python scripts/publication/verify_publication_data.py \\
    --experiment results/optA_20251012_143000 \\
    --paper luaran/JICEST_Paper.md

  # Strict mode (warnings become failures)
  python scripts/publication/verify_publication_data.py \\
    --experiment results/optA_20251012_143000 \\
    --strict
        """
    )

    parser.add_argument(
        '--experiment',
        required=True,
        help='Path to experiment results directory (use "results/optA_LATEST" for latest)'
    )

    parser.add_argument(
        '--paper',
        help='Path to paper markdown file to verify claims against'
    )

    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as failures'
    )

    parser.add_argument(
        '--output',
        help='Save report to file (optional)'
    )

    args = parser.parse_args()

    # Handle LATEST keyword
    experiment_dir = args.experiment
    if 'LATEST' in experiment_dir.upper():
        results_dir = Path('results')
        latest = find_latest_experiment(results_dir)
        if latest:
            experiment_dir = latest
            print(f"Using latest experiment: {latest}")
        else:
            print("ERROR: No experiments found in results directory")
            sys.exit(1)

    # Create verifier and run checks
    verifier = PublicationVerifier(
        experiment_dir=experiment_dir,
        paper_path=args.paper,
        strict=args.strict
    )

    # Run all verifications
    success = verifier.run_all_verifications()

    # Generate and print report
    report = verifier.generate_report()
    print("\n" + report)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()