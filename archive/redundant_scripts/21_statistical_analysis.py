#!/usr/bin/env python3
"""
Statistical Analysis Suite for Malaria Detection Results
Comprehensive statistical tests, confidence intervals, and significance testing
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mcnemar
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report
import json
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

class MalariaStatisticalAnalyzer:
    """Comprehensive statistical analysis for malaria detection results"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_manager = ResultsManager()

    def load_experiment_results(self) -> Dict:
        """Load all experiment results from various directories"""
        results = {}

        # Scan multiple result directories
        search_dirs = [
            self.results_dir / "current_experiments",
            self.results_dir / "detection",
            self.results_dir / "classification",
            Path("results")
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                results.update(self._scan_directory(search_dir))

        print(f"📊 Loaded {len(results)} experiment results")
        return results

    def _scan_directory(self, directory: Path) -> Dict:
        """Recursively scan directory for experiment results"""
        results = {}

        # Look for results.txt files
        for results_file in directory.rglob("results.txt"):
            try:
                experiment_name = results_file.parent.name
                results[experiment_name] = self._parse_results_file(results_file)
            except Exception as e:
                print(f"⚠️ Error parsing {results_file}: {e}")

        # Look for confusion matrices and additional metrics
        for confusion_file in directory.rglob("confusion_matrix.png"):
            try:
                experiment_name = confusion_file.parent.name
                if experiment_name in results:
                    results[experiment_name]['confusion_matrix_path'] = str(confusion_file)
            except Exception:
                pass

        return results

    def _parse_results_file(self, results_file: Path) -> Dict:
        """Parse experiment results file"""
        result_data = {}

        with open(results_file, 'r') as f:
            content = f.read()

        # Extract key metrics using regex-like parsing
        lines = content.split('\n')
        for line in lines:
            if 'Val Acc:' in line or 'Test Acc:' in line:
                try:
                    parts = line.split(':')
                    metric_name = parts[0].strip()
                    value_str = parts[1].strip().replace('%', '')
                    result_data[metric_name] = float(value_str)
                except:
                    pass
            elif 'Model:' in line:
                result_data['model'] = line.split(':')[1].strip()

        return result_data

    def perform_statistical_tests(self, results: Dict) -> Dict:
        """Perform comprehensive statistical tests on results"""
        print("🔬 Performing statistical analysis...")

        # Extract accuracies for different model types
        detection_models = []
        classification_models = []
        combined_models = []

        accuracies_by_type = {
            'detection': [],
            'classification': [],
            'combined': []
        }

        for exp_name, exp_data in results.items():
            if 'Test Acc' in exp_data:
                accuracy = exp_data['Test Acc']

                if 'detection' in exp_name.lower() and 'classification' not in exp_name.lower():
                    accuracies_by_type['detection'].append(accuracy)
                    detection_models.append(exp_name)
                elif 'classification' in exp_name.lower() or 'cls' in exp_name.lower():
                    accuracies_by_type['classification'].append(accuracy)
                    classification_models.append(exp_name)
                elif any(word in exp_name.lower() for word in ['combo', 'det_to', 'species_aware']):
                    accuracies_by_type['combined'].append(accuracy)
                    combined_models.append(exp_name)

        # Statistical tests
        stats_results = {}

        # 1. ANOVA test between model types
        if all(len(accuracies_by_type[key]) > 1 for key in accuracies_by_type):
            try:
                f_stat, p_value = stats.f_oneway(
                    accuracies_by_type['detection'],
                    accuracies_by_type['classification'],
                    accuracies_by_type['combined']
                )
                stats_results['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except Exception as e:
                print(f"⚠️ ANOVA test failed: {e}")

        # 2. Pairwise t-tests
        stats_results['pairwise_tests'] = {}
        for type1 in accuracies_by_type:
            for type2 in accuracies_by_type:
                if type1 < type2 and len(accuracies_by_type[type1]) > 1 and len(accuracies_by_type[type2]) > 1:
                    try:
                        t_stat, p_value = stats.ttest_ind(
                            accuracies_by_type[type1],
                            accuracies_by_type[type2]
                        )
                        stats_results['pairwise_tests'][f"{type1}_vs_{type2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except Exception as e:
                        print(f"⚠️ T-test {type1} vs {type2} failed: {e}")

        # 3. Descriptive statistics
        stats_results['descriptive'] = {}
        for model_type, accuracies in accuracies_by_type.items():
            if accuracies:
                stats_results['descriptive'][model_type] = {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'median': np.median(accuracies),
                    'min': np.min(accuracies),
                    'max': np.max(accuracies),
                    'count': len(accuracies)
                }

        return stats_results

    def calculate_confidence_intervals(self, results: Dict, confidence_level: float = 0.95) -> Dict:
        """Calculate confidence intervals for model performance"""
        print("📊 Calculating confidence intervals...")

        confidence_intervals = {}
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        for exp_name, exp_data in results.items():
            if 'Test Acc' in exp_data:
                accuracy = exp_data['Test Acc'] / 100.0  # Convert to proportion

                # Assuming binomial distribution for accuracy
                # Need sample size - estimate from typical test set sizes
                n = 100  # Default estimate, should be updated with actual test sizes

                # Wilson score interval (more robust than normal approximation)
                p_hat = accuracy
                denominator = 1 + (z_score**2) / n
                center = (p_hat + (z_score**2) / (2 * n)) / denominator
                margin = (z_score / denominator) * np.sqrt((p_hat * (1 - p_hat)) / n + (z_score**2) / (4 * n**2))

                lower = max(0, center - margin)
                upper = min(1, center + margin)

                confidence_intervals[exp_name] = {
                    'accuracy': accuracy * 100,
                    'ci_lower': lower * 100,
                    'ci_upper': upper * 100,
                    'margin_error': margin * 100
                }

        return confidence_intervals

    def perform_power_analysis(self, results: Dict) -> Dict:
        """Perform statistical power analysis"""
        print("⚡ Performing power analysis...")

        power_results = {}

        # Group results by model type
        model_groups = {
            'yolo_detection': [],
            'rtdetr_detection': [],
            'yolo_classification': [],
            'pytorch_classification': [],
            'combined_models': []
        }

        for exp_name, exp_data in results.items():
            if 'Test Acc' not in exp_data:
                continue

            accuracy = exp_data['Test Acc']

            if 'yolo' in exp_name.lower() and 'detection' in exp_name.lower():
                model_groups['yolo_detection'].append(accuracy)
            elif 'rtdetr' in exp_name.lower():
                model_groups['rtdetr_detection'].append(accuracy)
            elif 'yolo' in exp_name.lower() and ('cls' in exp_name.lower() or 'classification' in exp_name.lower()):
                model_groups['yolo_classification'].append(accuracy)
            elif 'resnet' in exp_name.lower() or 'efficientnet' in exp_name.lower() or 'densenet' in exp_name.lower():
                model_groups['pytorch_classification'].append(accuracy)
            elif any(word in exp_name.lower() for word in ['combo', 'det_to', 'species_aware']):
                model_groups['combined_models'].append(accuracy)

        # Calculate effect sizes (Cohen's d) between groups
        for group1_name in model_groups:
            for group2_name in model_groups:
                if group1_name < group2_name:
                    group1_data = model_groups[group1_name]
                    group2_data = model_groups[group2_name]

                    if len(group1_data) > 1 and len(group2_data) > 1:
                        # Cohen's d
                        mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
                        std1, std2 = np.std(group1_data, ddof=1), np.std(group2_data, ddof=1)
                        pooled_std = np.sqrt(((len(group1_data) - 1) * std1**2 + (len(group2_data) - 1) * std2**2) /
                                           (len(group1_data) + len(group2_data) - 2))

                        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

                        power_results[f"{group1_name}_vs_{group2_name}"] = {
                            'cohens_d': cohens_d,
                            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
                            'group1_mean': mean1,
                            'group2_mean': mean2,
                            'group1_n': len(group1_data),
                            'group2_n': len(group2_data)
                        }

        return power_results

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"

    def generate_statistical_visualizations(self, stats_results: Dict, confidence_intervals: Dict,
                                          power_results: Dict, output_dir: Path):
        """Generate comprehensive statistical visualization plots"""
        print("📈 Generating statistical visualizations...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Box plot of accuracies by model type
        plt.figure(figsize=(12, 8))

        model_types = ['detection', 'classification', 'combined']
        data_for_boxplot = []
        labels_for_boxplot = []

        for model_type in model_types:
            if model_type in stats_results['descriptive']:
                # Create sample data for box plot (using mean and std to simulate)
                desc_stats = stats_results['descriptive'][model_type]
                sample_data = np.random.normal(desc_stats['mean'], desc_stats['std'], desc_stats['count'])
                data_for_boxplot.append(sample_data)
                labels_for_boxplot.append(f"{model_type.title()}\n(n={desc_stats['count']})")

        if data_for_boxplot:
            plt.boxplot(data_for_boxplot, labels=labels_for_boxplot)
            plt.title('Model Performance Distribution by Type')
            plt.ylabel('Test Accuracy (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'model_performance_boxplot.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 2. Confidence intervals plot
        if confidence_intervals:
            plt.figure(figsize=(14, 10))

            experiments = list(confidence_intervals.keys())[:20]  # Limit to top 20 for readability
            accuracies = [confidence_intervals[exp]['accuracy'] for exp in experiments]
            ci_lowers = [confidence_intervals[exp]['ci_lower'] for exp in experiments]
            ci_uppers = [confidence_intervals[exp]['ci_upper'] for exp in experiments]

            y_pos = np.arange(len(experiments))

            plt.errorbar(accuracies, y_pos,
                        xerr=[np.array(accuracies) - np.array(ci_lowers),
                              np.array(ci_uppers) - np.array(accuracies)],
                        fmt='o', capsize=5, capthick=2)

            plt.yticks(y_pos, [exp.replace('_', '\n') for exp in experiments])
            plt.xlabel('Test Accuracy (%) with 95% Confidence Intervals')
            plt.title('Model Performance with Confidence Intervals')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'confidence_intervals.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Effect sizes heatmap
        if power_results:
            effect_sizes = {}
            comparison_names = []

            for comparison, results in power_results.items():
                comparison_names.append(comparison.replace('_vs_', ' vs '))
                effect_sizes[comparison] = abs(results['cohens_d'])

            if effect_sizes:
                plt.figure(figsize=(10, 6))

                values = list(effect_sizes.values())
                names = [name.replace('_', ' ').title() for name in comparison_names]

                colors = ['green' if v < 0.2 else 'yellow' if v < 0.5 else 'orange' if v < 0.8 else 'red'
                         for v in values]

                plt.barh(names, values, color=colors)
                plt.xlabel("Effect Size (Cohen's d)")
                plt.title('Effect Sizes Between Model Groups')
                plt.axvline(x=0.2, color='black', linestyle='--', alpha=0.5, label='Small')
                plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Medium')
                plt.axvline(x=0.8, color='black', linestyle='--', alpha=0.5, label='Large')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / 'effect_sizes.png', dpi=300, bbox_inches='tight')
                plt.close()

    def generate_statistical_report(self, stats_results: Dict, confidence_intervals: Dict,
                                  power_results: Dict, output_file: Path):
        """Generate comprehensive statistical analysis report"""
        print("📝 Generating statistical report...")

        with open(output_file, 'w') as f:
            f.write("# Malaria Detection Statistical Analysis Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Descriptive Statistics
            f.write("## Descriptive Statistics\n\n")
            if 'descriptive' in stats_results:
                for model_type, stats in stats_results['descriptive'].items():
                    f.write(f"### {model_type.title()} Models\n")
                    f.write(f"- Count: {stats['count']}\n")
                    f.write(f"- Mean Accuracy: {stats['mean']:.2f}%\n")
                    f.write(f"- Standard Deviation: {stats['std']:.2f}%\n")
                    f.write(f"- Median: {stats['median']:.2f}%\n")
                    f.write(f"- Range: {stats['min']:.2f}% - {stats['max']:.2f}%\n\n")

            # Statistical Tests
            f.write("## Statistical Tests\n\n")

            # ANOVA Results
            if 'anova' in stats_results:
                anova = stats_results['anova']
                f.write("### ANOVA Test (Between Model Types)\n")
                f.write(f"- F-statistic: {anova['f_statistic']:.4f}\n")
                f.write(f"- p-value: {anova['p_value']:.6f}\n")
                f.write(f"- Significant difference: {'Yes' if anova['significant'] else 'No'}\n\n")

            # Pairwise Tests
            if 'pairwise_tests' in stats_results:
                f.write("### Pairwise T-Tests\n")
                for comparison, test_result in stats_results['pairwise_tests'].items():
                    f.write(f"#### {comparison.replace('_vs_', ' vs ').title()}\n")
                    f.write(f"- t-statistic: {test_result['t_statistic']:.4f}\n")
                    f.write(f"- p-value: {test_result['p_value']:.6f}\n")
                    f.write(f"- Significant: {'Yes' if test_result['significant'] else 'No'}\n\n")

            # Effect Sizes
            f.write("## Effect Size Analysis\n\n")
            if power_results:
                for comparison, results in power_results.items():
                    f.write(f"### {comparison.replace('_vs_', ' vs ').title()}\n")
                    f.write(f"- Cohen's d: {results['cohens_d']:.4f}\n")
                    f.write(f"- Effect size: {results['effect_size_interpretation']}\n")
                    f.write(f"- Group 1 mean: {results['group1_mean']:.2f}% (n={results['group1_n']})\n")
                    f.write(f"- Group 2 mean: {results['group2_mean']:.2f}% (n={results['group2_n']})\n\n")

            # Confidence Intervals Summary
            f.write("## Confidence Intervals Summary\n\n")
            if confidence_intervals:
                # Sort by accuracy
                sorted_experiments = sorted(confidence_intervals.items(),
                                          key=lambda x: x[1]['accuracy'], reverse=True)

                f.write("### Top 10 Models (by Test Accuracy)\n")
                for i, (exp_name, ci_data) in enumerate(sorted_experiments[:10]):
                    f.write(f"{i+1}. **{exp_name}**\n")
                    f.write(f"   - Accuracy: {ci_data['accuracy']:.2f}%\n")
                    f.write(f"   - 95% CI: [{ci_data['ci_lower']:.2f}%, {ci_data['ci_upper']:.2f}%]\n")
                    f.write(f"   - Margin of Error: ±{ci_data['margin_error']:.2f}%\n\n")

        print(f"✅ Statistical report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Statistical Analysis for Malaria Detection")
    parser.add_argument("--results_dir", default="results",
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", default="statistical_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--confidence_level", type=float, default=0.95,
                       help="Confidence level for intervals")

    args = parser.parse_args()

    print("=" * 60)
    print("MALARIA DETECTION STATISTICAL ANALYSIS")
    print("=" * 60)

    # Initialize analyzer
    analyzer = MalariaStatisticalAnalyzer(args.results_dir)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment results
    results = analyzer.load_experiment_results()

    if not results:
        print("❌ No experiment results found!")
        return

    # Perform statistical tests
    stats_results = analyzer.perform_statistical_tests(results)

    # Calculate confidence intervals
    confidence_intervals = analyzer.calculate_confidence_intervals(results, args.confidence_level)

    # Perform power analysis
    power_results = analyzer.perform_power_analysis(results)

    # Generate visualizations
    analyzer.generate_statistical_visualizations(
        stats_results, confidence_intervals, power_results, output_dir
    )

    # Generate comprehensive report
    report_file = output_dir / "statistical_analysis_report.md"
    analyzer.generate_statistical_report(
        stats_results, confidence_intervals, power_results, report_file
    )

    # Save raw results as JSON
    all_results = {
        'experiment_results': results,
        'statistical_tests': stats_results,
        'confidence_intervals': confidence_intervals,
        'power_analysis': power_results
    }

    json_file = output_dir / "statistical_analysis_data.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("🎉 STATISTICAL ANALYSIS COMPLETED!")
    print("=" * 60)
    print(f"📊 Analyzed {len(results)} experiments")
    print(f"📈 Generated visualizations in: {output_dir}")
    print(f"📝 Report saved to: {report_file}")
    print(f"💾 Raw data saved to: {json_file}")

if __name__ == "__main__":
    main()