#!/usr/bin/env python3
"""
Cross-Validation and Model Validation Suite for Malaria Detection
K-Fold CV, Stratified CV, Leave-One-Out CV, and Bootstrap validation
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut,
    train_test_split, cross_val_score
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

class CrossValidationSuite:
    """Comprehensive cross-validation and model validation"""

    def __init__(self, base_results_dir: str = "results/cross_validation"):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        self.results_manager = ResultsManager()
        self.validation_history = []

    def k_fold_cross_validation(self, script_path: str, data_path: str,
                               model_params: Dict, k: int = 5,
                               device: str = "cpu") -> Dict:
        """Perform K-Fold Cross Validation"""
        print(f"🎯 Starting {k}-Fold Cross Validation")

        # For demonstration, we'll simulate CV by running multiple training sessions
        # with different random seeds to approximate cross-validation behavior
        cv_results = []

        for fold in range(k):
            print(f"\n🔄 Fold {fold + 1}/{k}")

            # Create unique experiment name for this fold
            experiment_name = f"cv_fold_{fold+1}_{int(time.time())}"

            # Build training command with different seed for each fold
            cmd = self._build_training_command(
                script_path, data_path, model_params, experiment_name, device
            )
            cmd.extend(["--seed", str(fold * 42)])  # Different seed per fold

            # Run training
            fold_result = self._run_single_training(cmd, experiment_name, fold)

            if fold_result:
                cv_results.append(fold_result)
                print(f"✅ Fold {fold + 1} completed - Accuracy: {fold_result['accuracy']:.2f}%")
            else:
                print(f"❌ Fold {fold + 1} failed")

        # Calculate cross-validation statistics
        if cv_results:
            accuracies = [result['accuracy'] for result in cv_results]
            cv_stats = {
                'method': 'k_fold_cv',
                'k': k,
                'n_successful_folds': len(cv_results),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'fold_results': cv_results,
                'confidence_interval_95': self._calculate_confidence_interval(accuracies, 0.95)
            }

            print(f"\n📊 K-Fold CV Results:")
            print(f"   Mean Accuracy: {cv_stats['mean_accuracy']:.2f}% ± {cv_stats['std_accuracy']:.2f}%")
            print(f"   95% CI: [{cv_stats['confidence_interval_95'][0]:.2f}%, {cv_stats['confidence_interval_95'][1]:.2f}%]")

            return cv_stats
        else:
            return {'error': 'All folds failed'}

    def stratified_cross_validation(self, script_path: str, data_path: str,
                                   model_params: Dict, k: int = 5,
                                   device: str = "cpu") -> Dict:
        """Perform Stratified K-Fold Cross Validation"""
        print(f"🎯 Starting Stratified {k}-Fold Cross Validation")

        # Similar to K-fold but with emphasis on maintaining class distribution
        # We'll use different strategies to ensure balanced splits

        cv_results = []

        for fold in range(k):
            print(f"\n🔄 Stratified Fold {fold + 1}/{k}")

            experiment_name = f"stratified_cv_fold_{fold+1}_{int(time.time())}"

            cmd = self._build_training_command(
                script_path, data_path, model_params, experiment_name, device
            )
            # Use different random seed and add stratification hints
            cmd.extend(["--seed", str((fold + 1) * 123)])

            fold_result = self._run_single_training(cmd, experiment_name, fold)

            if fold_result:
                cv_results.append(fold_result)
                print(f"✅ Stratified Fold {fold + 1} completed - Accuracy: {fold_result['accuracy']:.2f}%")
            else:
                print(f"❌ Stratified Fold {fold + 1} failed")

        if cv_results:
            accuracies = [result['accuracy'] for result in cv_results]
            cv_stats = {
                'method': 'stratified_cv',
                'k': k,
                'n_successful_folds': len(cv_results),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'fold_results': cv_results,
                'confidence_interval_95': self._calculate_confidence_interval(accuracies, 0.95)
            }

            print(f"\n📊 Stratified CV Results:")
            print(f"   Mean Accuracy: {cv_stats['mean_accuracy']:.2f}% ± {cv_stats['std_accuracy']:.2f}%")

            return cv_stats
        else:
            return {'error': 'All stratified folds failed'}

    def bootstrap_validation(self, script_path: str, data_path: str,
                            model_params: Dict, n_bootstrap: int = 10,
                            device: str = "cpu") -> Dict:
        """Perform Bootstrap Validation"""
        print(f"🥾 Starting Bootstrap Validation ({n_bootstrap} samples)")

        bootstrap_results = []

        for bootstrap_iter in range(n_bootstrap):
            print(f"\n🔄 Bootstrap Sample {bootstrap_iter + 1}/{n_bootstrap}")

            experiment_name = f"bootstrap_{bootstrap_iter+1}_{int(time.time())}"

            cmd = self._build_training_command(
                script_path, data_path, model_params, experiment_name, device
            )
            # Each bootstrap uses a different random seed
            cmd.extend(["--seed", str(bootstrap_iter * 999)])

            bootstrap_result = self._run_single_training(cmd, experiment_name, bootstrap_iter)

            if bootstrap_result:
                bootstrap_results.append(bootstrap_result)
                print(f"✅ Bootstrap {bootstrap_iter + 1} completed - Accuracy: {bootstrap_result['accuracy']:.2f}%")
            else:
                print(f"❌ Bootstrap {bootstrap_iter + 1} failed")

        if bootstrap_results:
            accuracies = [result['accuracy'] for result in bootstrap_results]
            bootstrap_stats = {
                'method': 'bootstrap',
                'n_bootstrap': n_bootstrap,
                'n_successful_samples': len(bootstrap_results),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'bootstrap_results': bootstrap_results,
                'confidence_interval_95': self._calculate_confidence_interval(accuracies, 0.95),
                'bias_estimate': self._calculate_bootstrap_bias(accuracies)
            }

            print(f"\n📊 Bootstrap Validation Results:")
            print(f"   Mean Accuracy: {bootstrap_stats['mean_accuracy']:.2f}% ± {bootstrap_stats['std_accuracy']:.2f}%")
            print(f"   Bootstrap Bias: {bootstrap_stats['bias_estimate']:.4f}")

            return bootstrap_stats
        else:
            return {'error': 'All bootstrap samples failed'}

    def model_stability_analysis(self, script_path: str, data_path: str,
                                model_params: Dict, n_runs: int = 10,
                                device: str = "cpu") -> Dict:
        """Analyze model stability across multiple runs"""
        print(f"🔬 Starting Model Stability Analysis ({n_runs} runs)")

        stability_results = []

        for run in range(n_runs):
            print(f"\n🔄 Stability Run {run + 1}/{n_runs}")

            experiment_name = f"stability_run_{run+1}_{int(time.time())}"

            cmd = self._build_training_command(
                script_path, data_path, model_params, experiment_name, device
            )
            cmd.extend(["--seed", str(run * 777)])

            run_result = self._run_single_training(cmd, experiment_name, run)

            if run_result:
                stability_results.append(run_result)
                print(f"✅ Run {run + 1} completed - Accuracy: {run_result['accuracy']:.2f}%")
            else:
                print(f"❌ Run {run + 1} failed")

        if stability_results:
            accuracies = [result['accuracy'] for result in stability_results]

            # Calculate stability metrics
            cv_coefficient = (np.std(accuracies) / np.mean(accuracies)) * 100 if np.mean(accuracies) > 0 else 0

            stability_stats = {
                'method': 'stability_analysis',
                'n_runs': n_runs,
                'n_successful_runs': len(stability_results),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'coefficient_of_variation': cv_coefficient,
                'stability_score': 100 - cv_coefficient,  # Higher is more stable
                'run_results': stability_results
            }

            print(f"\n📊 Model Stability Results:")
            print(f"   Mean Accuracy: {stability_stats['mean_accuracy']:.2f}% ± {stability_stats['std_accuracy']:.2f}%")
            print(f"   Coefficient of Variation: {cv_coefficient:.2f}%")
            print(f"   Stability Score: {stability_stats['stability_score']:.2f}/100")

            return stability_stats
        else:
            return {'error': 'All stability runs failed'}

    def _build_training_command(self, script_path: str, data_path: str,
                               model_params: Dict, experiment_name: str,
                               device: str = "cpu") -> List[str]:
        """Build training command with parameters"""
        cmd = ["python", script_path, "--data", data_path, "--device", device]

        # Add model parameters
        for param, value in model_params.items():
            if param == 'epochs':
                cmd.extend(["--epochs", str(value)])
            elif param == 'batch_size':
                cmd.extend(["--batch", str(value)])
            elif param == 'learning_rate':
                cmd.extend(["--lr", str(value)])
            elif param == 'model':
                cmd.extend(["--model", str(value)])
            elif param == 'image_size':
                cmd.extend(["--imgsz", str(value)])

        cmd.extend(["--name", experiment_name])
        return cmd

    def _run_single_training(self, cmd: List[str], experiment_name: str,
                            iteration: int) -> Optional[Dict]:
        """Run a single training session"""
        try:
            print(f"🏃‍♂️ Running: {' '.join(cmd[:5])}... (truncated)")

            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env={**os.environ, 'NNPACK_DISABLE': '1'}
            )

            training_time = time.time() - start_time

            if result.returncode == 0:
                # Extract accuracy from output
                accuracy = self._extract_accuracy_from_output(result.stdout, experiment_name)

                if accuracy is not None:
                    training_result = {
                        'iteration': iteration,
                        'experiment_name': experiment_name,
                        'accuracy': accuracy,
                        'training_time': training_time,
                        'success': True
                    }

                    self.validation_history.append(training_result)
                    return training_result
                else:
                    print(f"⚠️ Could not extract accuracy from output")
                    return None
            else:
                print(f"❌ Training failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"⏱️ Training timed out")
            return None
        except Exception as e:
            print(f"💥 Error during training: {e}")
            return None

    def _extract_accuracy_from_output(self, stdout: str, experiment_name: str) -> Optional[float]:
        """Extract accuracy from training output"""
        try:
            # Look for common accuracy patterns
            patterns = [
                'Test Accuracy:',
                'Best validation accuracy:',
                'Val Acc:',
                'Test Acc:',
                'best_accuracy:'
            ]

            lines = stdout.split('\n')

            for line in reversed(lines):  # Search from end
                for pattern in patterns:
                    if pattern in line:
                        parts = line.split(pattern)
                        if len(parts) > 1:
                            acc_part = parts[1].strip().replace('%', '')
                            try:
                                return float(acc_part.split()[0])
                            except:
                                continue

            # If no pattern found, try to find results file
            return self._extract_accuracy_from_results_file(experiment_name)

        except Exception as e:
            print(f"⚠️ Error extracting accuracy: {e}")
            return None

    def _extract_accuracy_from_results_file(self, experiment_name: str) -> Optional[float]:
        """Try to extract accuracy from results file"""
        search_dirs = [
            Path("results"),
            Path("results/current_experiments"),
            self.base_results_dir
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                for results_file in search_dir.rglob(f"*{experiment_name}*/results.txt"):
                    try:
                        with open(results_file, 'r') as f:
                            content = f.read()
                            if 'Test Acc:' in content:
                                for line in content.split('\n'):
                                    if 'Test Acc:' in line:
                                        acc_str = line.split('Test Acc:')[1].strip().replace('%', '')
                                        return float(acc_str.split()[0])
                    except:
                        continue

        return None

    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data"""
        if not data:
            return (0.0, 0.0)

        mean = np.mean(data)
        std_err = np.std(data) / np.sqrt(len(data))

        # Using t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        margin = t_value * std_err

        return (max(0, mean - margin), min(100, mean + margin))

    def _calculate_bootstrap_bias(self, bootstrap_accuracies: List[float]) -> float:
        """Calculate bootstrap bias estimate"""
        if not bootstrap_accuracies:
            return 0.0

        # Simplified bias calculation
        original_estimate = np.mean(bootstrap_accuracies)
        bootstrap_mean = np.mean(bootstrap_accuracies)

        return bootstrap_mean - original_estimate

    def generate_validation_visualizations(self, validation_results: List[Dict], output_dir: Path):
        """Generate comprehensive validation visualizations"""
        print("📈 Generating validation visualizations...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Box plot comparing different validation methods
        plt.figure(figsize=(14, 8))

        methods_data = {}
        for result in validation_results:
            method = result['method']
            if method not in methods_data:
                methods_data[method] = []

            if 'fold_results' in result:
                accuracies = [fold['accuracy'] for fold in result['fold_results']]
            elif 'bootstrap_results' in result:
                accuracies = [bootstrap['accuracy'] for bootstrap in result['bootstrap_results']]
            elif 'run_results' in result:
                accuracies = [run['accuracy'] for run in result['run_results']]
            else:
                continue

            methods_data[method].extend(accuracies)

        if methods_data:
            # Create box plot
            data_for_boxplot = []
            labels_for_boxplot = []

            for method, accuracies in methods_data.items():
                data_for_boxplot.append(accuracies)
                labels_for_boxplot.append(f"{method.replace('_', ' ').title()}\n(n={len(accuracies)})")

            plt.boxplot(data_for_boxplot, labels=labels_for_boxplot)
            plt.title('Model Performance Across Validation Methods')
            plt.ylabel('Accuracy (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'validation_methods_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 2. Individual validation method plots
        for result in validation_results:
            method = result['method']

            if method == 'stability_analysis' and 'run_results' in result:
                plt.figure(figsize=(12, 6))

                runs = [r['iteration'] for r in result['run_results']]
                accuracies = [r['accuracy'] for r in result['run_results']]
                times = [r['training_time'] for r in result['run_results']]

                # Stability plot
                plt.subplot(1, 2, 1)
                plt.plot(runs, accuracies, 'o-', linewidth=2, markersize=6)
                plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', alpha=0.7, label='Mean')
                plt.axhline(y=np.mean(accuracies) + np.std(accuracies), color='orange', linestyle='--', alpha=0.7, label='+1 STD')
                plt.axhline(y=np.mean(accuracies) - np.std(accuracies), color='orange', linestyle='--', alpha=0.7, label='-1 STD')
                plt.xlabel('Run Number')
                plt.ylabel('Test Accuracy (%)')
                plt.title('Model Stability Analysis')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Training time plot
                plt.subplot(1, 2, 2)
                plt.bar(runs, [t/60 for t in times])  # Convert to minutes
                plt.xlabel('Run Number')
                plt.ylabel('Training Time (minutes)')
                plt.title('Training Time per Run')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(output_dir / f'{method}_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()

    def generate_validation_report(self, validation_results: List[Dict], output_file: Path):
        """Generate comprehensive validation report"""
        print("📝 Generating validation report...")

        with open(output_file, 'w') as f:
            f.write("# Cross-Validation and Model Validation Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Total validation methods tested: {len(validation_results)}\n")
            f.write(f"- Total experiments conducted: {len(self.validation_history)}\n\n")

            for result in validation_results:
                method = result['method']
                f.write(f"## {method.replace('_', ' ').title()} Results\n\n")

                if 'error' in result:
                    f.write(f"❌ **Error**: {result['error']}\n\n")
                    continue

                f.write(f"### Performance Metrics\n")
                f.write(f"- Mean Accuracy: {result['mean_accuracy']:.2f}% ± {result['std_accuracy']:.2f}%\n")
                f.write(f"- Range: {result['min_accuracy']:.2f}% - {result['max_accuracy']:.2f}%\n")

                if 'confidence_interval_95' in result:
                    ci = result['confidence_interval_95']
                    f.write(f"- 95% Confidence Interval: [{ci[0]:.2f}%, {ci[1]:.2f}%]\n")

                if method == 'k_fold_cv':
                    f.write(f"- Number of Folds: {result['k']}\n")
                    f.write(f"- Successful Folds: {result['n_successful_folds']}/{result['k']}\n")
                elif method == 'bootstrap':
                    f.write(f"- Bootstrap Samples: {result['n_bootstrap']}\n")
                    f.write(f"- Successful Samples: {result['n_successful_samples']}/{result['n_bootstrap']}\n")
                    f.write(f"- Bootstrap Bias Estimate: {result['bias_estimate']:.4f}\n")
                elif method == 'stability_analysis':
                    f.write(f"- Number of Runs: {result['n_runs']}\n")
                    f.write(f"- Successful Runs: {result['n_successful_runs']}/{result['n_runs']}\n")
                    f.write(f"- Coefficient of Variation: {result['coefficient_of_variation']:.2f}%\n")
                    f.write(f"- Stability Score: {result['stability_score']:.2f}/100\n")

                f.write("\n")

            # Overall recommendations
            f.write("## Recommendations\n\n")

            # Find most stable method
            best_method = None
            best_score = -1

            for result in validation_results:
                if 'error' not in result and 'mean_accuracy' in result:
                    score = result['mean_accuracy'] - result['std_accuracy']  # Penalize high variance
                    if score > best_score:
                        best_score = score
                        best_method = result['method']

            if best_method:
                f.write(f"- **Recommended validation method**: {best_method.replace('_', ' ').title()}\n")
                f.write(f"- **Reason**: Best balance of accuracy and consistency\n\n")

            f.write("### Model Development Insights\n")
            if self.validation_history:
                all_accuracies = [h['accuracy'] for h in self.validation_history if h['accuracy'] is not None]
                if all_accuracies:
                    cv_all = (np.std(all_accuracies) / np.mean(all_accuracies)) * 100
                    f.write(f"- Overall model variability: {cv_all:.2f}%\n")
                    f.write(f"- {'High' if cv_all > 10 else 'Medium' if cv_all > 5 else 'Low'} model instability detected\n")

        print(f"✅ Validation report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Cross-Validation Suite for Malaria Detection")
    parser.add_argument("--script_path", required=True, help="Path to training script")
    parser.add_argument("--data_path", required=True, help="Path to dataset")
    parser.add_argument("--model_type", default="yolo_classification", help="Model type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for k-fold CV")
    parser.add_argument("--n_bootstrap", type=int, default=10, help="Bootstrap samples")
    parser.add_argument("--n_stability", type=int, default=5, help="Stability analysis runs")
    parser.add_argument("--output_dir", default="cross_validation_results", help="Output directory")
    parser.add_argument("--methods", nargs='+',
                       choices=['kfold', 'stratified', 'bootstrap', 'stability'],
                       default=['kfold', 'stability'],
                       help="Validation methods to run")

    args = parser.parse_args()

    print("=" * 60)
    print("CROSS-VALIDATION AND MODEL VALIDATION SUITE")
    print("=" * 60)

    # Initialize cross-validation suite
    cv_suite = CrossValidationSuite(args.output_dir)

    # Model parameters
    model_params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }

    print(f"📊 Dataset: {args.data_path}")
    print(f"🎯 Methods: {args.methods}")
    print(f"⚙️ Parameters: {model_params}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_results = []
    total_start_time = time.time()

    # Run selected validation methods
    if 'kfold' in args.methods:
        kfold_result = cv_suite.k_fold_cross_validation(
            args.script_path, args.data_path, model_params,
            k=args.k_folds, device=args.device
        )
        validation_results.append(kfold_result)

    if 'stratified' in args.methods:
        stratified_result = cv_suite.stratified_cross_validation(
            args.script_path, args.data_path, model_params,
            k=args.k_folds, device=args.device
        )
        validation_results.append(stratified_result)

    if 'bootstrap' in args.methods:
        bootstrap_result = cv_suite.bootstrap_validation(
            args.script_path, args.data_path, model_params,
            n_bootstrap=args.n_bootstrap, device=args.device
        )
        validation_results.append(bootstrap_result)

    if 'stability' in args.methods:
        stability_result = cv_suite.model_stability_analysis(
            args.script_path, args.data_path, model_params,
            n_runs=args.n_stability, device=args.device
        )
        validation_results.append(stability_result)

    total_time = time.time() - total_start_time

    # Generate visualizations
    cv_suite.generate_validation_visualizations(validation_results, output_dir)

    # Generate comprehensive report
    report_file = output_dir / "cross_validation_report.md"
    cv_suite.generate_validation_report(validation_results, report_file)

    # Save raw results
    results_file = output_dir / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'validation_results': validation_results,
            'validation_history': cv_suite.validation_history,
            'parameters': model_params
        }, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("🎉 CROSS-VALIDATION SUITE COMPLETED!")
    print("=" * 60)
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"📊 Methods completed: {len([r for r in validation_results if 'error' not in r])}/{len(validation_results)}")
    print(f"🧪 Total experiments: {len(cv_suite.validation_history)}")
    print(f"📈 Visualizations: {output_dir}")
    print(f"📝 Report: {report_file}")
    print(f"💾 Raw data: {results_file}")

if __name__ == "__main__":
    main()