#!/usr/bin/env python3
"""
Comprehensive Comparison Analysis: Detection vs Ground Truth Classification
This script compares classification performance between models trained on:
1. YOLOv8 detection-generated crops
2. YOLOv11 detection-generated crops
3. Ground truth annotation-generated crops
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
import re
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

def find_completed_experiments():
    """Find all completed classification experiments"""
    results_manager = ResultsManager()
    base_path = results_manager.base_dir

    experiments = {
        'yolo8_detection': [],
        'yolo11_detection': [],
        'ground_truth': []
    }

    # Define experiment patterns for each source
    patterns = {
        'yolo8_detection': ['*yolo8_det_to_*', '*yolo8_detection_*'],
        'yolo11_detection': ['*yolo11_det_to_*', '*yolo11_detection_*'],
        'ground_truth': ['*ground_truth_to_*', '*ground_truth_*']
    }

    # Search for experiment directories
    training_path = base_path / "current_experiments" / "training"
    if training_path.exists():
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = glob.glob(str(training_path / "**" / pattern), recursive=True)
                for match in matches:
                    exp_path = Path(match)
                    if exp_path.is_dir():
                        # Check if experiment has results
                        if (exp_path / "results.txt").exists() or (exp_path / "best.pt").exists():
                            experiments[category].append({
                                'path': exp_path,
                                'name': exp_path.name,
                                'category': category
                            })

    return experiments

def extract_results_from_experiment(exp_path):
    """Extract key metrics from experiment results"""
    results = {
        'experiment_name': exp_path.name,
        'path': str(exp_path),
        'status': 'unknown'
    }

    # Try to read results.txt file
    results_file = exp_path / "results.txt"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                content = f.read()

                # Extract key metrics using regex
                metrics = {}

                # Look for accuracy patterns
                acc_patterns = [
                    r'Test Acc(?:uracy)?:?\s*([0-9]+\.?[0-9]*)%?',
                    r'Val Acc(?:uracy)?:?\s*([0-9]+\.?[0-9]*)%?',
                    r'Best Val Acc:?\s*([0-9]+\.?[0-9]*)%?',
                    r'Top-1 Accuracy:?\s*([0-9]+\.?[0-9]*)',
                    r'accuracy:\s*([0-9]+\.?[0-9]*)'
                ]

                for pattern in acc_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        metrics['accuracy'] = float(match.group(1))
                        break

                # Look for loss patterns
                loss_patterns = [
                    r'Test Loss:?\s*([0-9]+\.?[0-9]*)',
                    r'Val Loss:?\s*([0-9]+\.?[0-9]*)',
                    r'loss:\s*([0-9]+\.?[0-9]*)'
                ]

                for pattern in loss_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        metrics['loss'] = float(match.group(1))
                        break

                # Look for model info
                model_match = re.search(r'Model:?\s*([^\n]+)', content, re.IGNORECASE)
                if model_match:
                    metrics['model'] = model_match.group(1).strip()

                # Look for training time
                time_patterns = [
                    r'Training Time:?\s*([0-9]+\.?[0-9]*)\s*min',
                    r'training time:?\s*([0-9]+\.?[0-9]*)\s*minutes?'
                ]

                for pattern in time_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        metrics['training_time_min'] = float(match.group(1))
                        break

                results.update(metrics)
                results['status'] = 'completed'

        except Exception as e:
            print(f"Error reading results from {results_file}: {e}")

    # Check for model weights
    if (exp_path / "best.pt").exists():
        results['has_weights'] = True

    # Check for plots
    if (exp_path / "confusion_matrix.png").exists():
        results['has_confusion_matrix'] = True

    if (exp_path / "training_curves.png").exists():
        results['has_training_curves'] = True

    return results

def load_crop_metadata():
    """Load metadata about the generated crops"""
    metadata = {}

    # Load YOLOv8 detection crops metadata
    yolo8_meta_path = Path("data/crops_from_yolo8_detection/crop_metadata.csv")
    if yolo8_meta_path.exists():
        metadata['yolo8_detection'] = pd.read_csv(yolo8_meta_path)

    # Load YOLOv11 detection crops metadata
    yolo11_meta_path = Path("data/crops_from_yolo11_detection/crop_metadata.csv")
    if yolo11_meta_path.exists():
        metadata['yolo11_detection'] = pd.read_csv(yolo11_meta_path)

    # Load ground truth crops metadata
    gt_meta_path = Path("data/crops_from_ground_truth/ground_truth_crop_metadata.csv")
    if gt_meta_path.exists():
        metadata['ground_truth'] = pd.read_csv(gt_meta_path)

    return metadata

def analyze_crop_quality(metadata):
    """Analyze the quality and distribution of generated crops"""
    analysis = {}

    for source, df in metadata.items():
        if df is not None and len(df) > 0:
            analysis[source] = {
                'total_crops': len(df),
                'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else None,
                'min_confidence': df['confidence'].min() if 'confidence' in df.columns else None,
                'max_confidence': df['confidence'].max() if 'confidence' in df.columns else None,
                'split_distribution': df['split'].value_counts().to_dict() if 'split' in df.columns else {},
                'unique_source_images': df['original_image'].nunique() if 'original_image' in df.columns else None
            }

            # Calculate crops per image
            if 'original_image' in df.columns:
                crops_per_image = df.groupby('original_image').size()
                analysis[source]['avg_crops_per_image'] = crops_per_image.mean()
                analysis[source]['max_crops_per_image'] = crops_per_image.max()
                analysis[source]['min_crops_per_image'] = crops_per_image.min()

    return analysis

def create_comparison_plots(results_df, crop_analysis, output_dir):
    """Create comprehensive comparison plots"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Accuracy comparison by source and model
    if 'accuracy' in results_df.columns and 'source' in results_df.columns:
        plt.figure(figsize=(12, 8))

        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Accuracy by source
        if len(results_df['source'].unique()) > 1:
            sns.boxplot(data=results_df, x='source', y='accuracy', ax=axes[0,0])
            axes[0,0].set_title('Classification Accuracy by Crop Source')
            axes[0,0].set_ylabel('Accuracy (%)')
            axes[0,0].tick_params(axis='x', rotation=45)

        # Plot 2: Accuracy by model type
        if 'model_type' in results_df.columns:
            sns.boxplot(data=results_df, x='model_type', y='accuracy', ax=axes[0,1])
            axes[0,1].set_title('Classification Accuracy by Model Type')
            axes[0,1].set_ylabel('Accuracy (%)')

        # Plot 3: Heatmap of source vs model performance
        if 'model_type' in results_df.columns and len(results_df) > 0:
            pivot_df = results_df.pivot_table(values='accuracy', index='source', columns='model_type', aggfunc='mean')
            sns.heatmap(pivot_df, annot=True, fmt='.2f', ax=axes[1,0], cmap='viridis')
            axes[1,0].set_title('Average Accuracy: Source vs Model Type')

        # Plot 4: Training time comparison
        if 'training_time_min' in results_df.columns:
            sns.barplot(data=results_df, x='source', y='training_time_min', ax=axes[1,1])
            axes[1,1].set_title('Training Time by Crop Source')
            axes[1,1].set_ylabel('Training Time (minutes)')
            axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / 'classification_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Crop quality analysis plots
    if crop_analysis:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot crop counts
        sources = list(crop_analysis.keys())
        crop_counts = [crop_analysis[source]['total_crops'] for source in sources]

        axes[0,0].bar(sources, crop_counts)
        axes[0,0].set_title('Total Crops Generated by Source')
        axes[0,0].set_ylabel('Number of Crops')
        axes[0,0].tick_params(axis='x', rotation=45)

        # Plot confidence distribution (only for detection sources)
        conf_sources = []
        avg_confidences = []
        for source in sources:
            if crop_analysis[source]['avg_confidence'] is not None:
                conf_sources.append(source)
                avg_confidences.append(crop_analysis[source]['avg_confidence'])

        if conf_sources:
            axes[0,1].bar(conf_sources, avg_confidences)
            axes[0,1].set_title('Average Detection Confidence')
            axes[0,1].set_ylabel('Confidence')
            axes[0,1].tick_params(axis='x', rotation=45)

        # Plot crops per image
        cpi_sources = []
        avg_crops_per_image = []
        for source in sources:
            if crop_analysis[source].get('avg_crops_per_image') is not None:
                cpi_sources.append(source)
                avg_crops_per_image.append(crop_analysis[source]['avg_crops_per_image'])

        if cpi_sources:
            axes[1,0].bar(cpi_sources, avg_crops_per_image)
            axes[1,0].set_title('Average Crops per Source Image')
            axes[1,0].set_ylabel('Crops per Image')
            axes[1,0].tick_params(axis='x', rotation=45)

        # Plot split distribution
        if sources:
            split_data = []
            for source in sources:
                splits = crop_analysis[source]['split_distribution']
                for split, count in splits.items():
                    split_data.append({'source': source, 'split': split, 'count': count})

            if split_data:
                split_df = pd.DataFrame(split_data)
                sns.barplot(data=split_df, x='source', y='count', hue='split', ax=axes[1,1])
                axes[1,1].set_title('Crop Distribution by Split')
                axes[1,1].set_ylabel('Number of Crops')
                axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / 'crop_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_comparison_report(results_df, crop_analysis, output_dir):
    """Generate detailed comparison report"""
    output_path = Path(output_dir)
    report_file = output_path / 'comparison_report.md'

    with open(report_file, 'w') as f:
        f.write("# Detection vs Ground Truth Classification Comparison Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        if len(results_df) > 0:
            f.write(f"- **Total Experiments Analyzed**: {len(results_df)}\n")
            f.write(f"- **Crop Sources**: {', '.join(results_df['source'].unique())}\n")
            if 'model_type' in results_df.columns:
                f.write(f"- **Model Types**: {', '.join(results_df['model_type'].unique())}\n")

            if 'accuracy' in results_df.columns:
                best_acc = results_df.loc[results_df['accuracy'].idxmax()]
                f.write(f"- **Best Performance**: {best_acc['accuracy']:.2f}% ({best_acc['experiment_name']})\n")

                # Performance by source
                source_perf = results_df.groupby('source')['accuracy'].agg(['mean', 'std', 'count'])
                f.write("\n### Performance by Crop Source:\n")
                for source, stats in source_perf.iterrows():
                    f.write(f"- **{source}**: {stats['mean']:.2f}% ± {stats['std']:.2f}% ({stats['count']} experiments)\n")

        # Crop Analysis
        f.write("\n## Crop Generation Analysis\n\n")
        for source, analysis in crop_analysis.items():
            f.write(f"### {source.replace('_', ' ').title()}\n")
            f.write(f"- **Total Crops**: {analysis['total_crops']:,}\n")
            f.write(f"- **Source Images**: {analysis.get('unique_source_images', 'N/A')}\n")
            f.write(f"- **Avg Crops per Image**: {analysis.get('avg_crops_per_image', 'N/A'):.2f}\n")

            if analysis['avg_confidence'] is not None:
                f.write(f"- **Avg Detection Confidence**: {analysis['avg_confidence']:.3f}\n")
                f.write(f"- **Confidence Range**: {analysis['min_confidence']:.3f} - {analysis['max_confidence']:.3f}\n")

            f.write(f"- **Split Distribution**:\n")
            for split, count in analysis['split_distribution'].items():
                f.write(f"  - {split}: {count:,} crops\n")
            f.write("\n")

        # Detailed Results
        f.write("## Detailed Experiment Results\n\n")
        f.write("| Experiment | Source | Model | Accuracy | Loss | Training Time |\n")
        f.write("|------------|--------|-------|----------|------|---------------|\n")

        for _, row in results_df.iterrows():
            acc = f"{row.get('accuracy', 'N/A'):.2f}%" if pd.notna(row.get('accuracy')) else "N/A"
            loss = f"{row.get('loss', 'N/A'):.4f}" if pd.notna(row.get('loss')) else "N/A"
            time_str = f"{row.get('training_time_min', 'N/A'):.1f} min" if pd.notna(row.get('training_time_min')) else "N/A"
            model = row.get('model', 'N/A')

            f.write(f"| {row['experiment_name']} | {row['source']} | {model} | {acc} | {loss} | {time_str} |\n")

        # Conclusions
        f.write("\n## Key Findings\n\n")

        if len(results_df) > 0 and 'accuracy' in results_df.columns:
            # Best source
            source_means = results_df.groupby('source')['accuracy'].mean()
            best_source = source_means.idxmax()
            f.write(f"1. **Best Crop Source**: {best_source.replace('_', ' ').title()} achieved highest average accuracy ({source_means[best_source]:.2f}%)\n")

            # Model comparison
            if 'model_type' in results_df.columns:
                model_means = results_df.groupby('model_type')['accuracy'].mean()
                best_model = model_means.idxmax()
                f.write(f"2. **Best Model Type**: {best_model} achieved highest average accuracy ({model_means[best_model]:.2f}%)\n")

            # Crop quality impact
            total_crops = sum(analysis['total_crops'] for analysis in crop_analysis.values())
            f.write(f"3. **Crop Generation**: Generated {total_crops:,} total crops across all sources\n")

            # Quality vs quantity tradeoff
            if 'yolo8_detection' in crop_analysis and 'ground_truth' in crop_analysis:
                yolo8_crops = crop_analysis['yolo8_detection']['total_crops']
                gt_crops = crop_analysis['ground_truth']['total_crops']
                f.write(f"4. **Detection vs Ground Truth**: YOLOv8 generated {yolo8_crops:,} crops vs {gt_crops:,} from ground truth\n")

        f.write("\n## Recommendations\n\n")
        f.write("1. Use the best-performing crop source and model combination for production\n")
        f.write("2. Consider the trade-off between crop quantity (detection) and quality (ground truth)\n")
        f.write("3. Investigate failure cases in lower-performing combinations\n")
        f.write("4. Consider ensemble methods combining multiple approaches\n")

def main():
    parser = argparse.ArgumentParser(description="Compare Detection vs Ground Truth Classification Performance")
    parser.add_argument("--output", default="results/comparison_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--include_running", action="store_true",
                       help="Include running experiments in analysis")

    args = parser.parse_args()

    print("=" * 80)
    print("DETECTION vs GROUND TRUTH CLASSIFICATION COMPARISON")
    print("=" * 80)

    # Find completed experiments
    print("🔍 Finding completed experiments...")
    experiments = find_completed_experiments()

    # Extract results from each experiment
    all_results = []
    for source, exp_list in experiments.items():
        print(f"\n📊 Analyzing {source} experiments ({len(exp_list)} found)...")
        for exp in exp_list:
            results = extract_results_from_experiment(exp['path'])
            results['source'] = source

            # Determine model type from experiment name
            exp_name = exp['name'].lower()
            if 'yolo' in exp_name:
                results['model_type'] = 'YOLO'
            elif 'resnet' in exp_name:
                results['model_type'] = 'ResNet'
            elif 'efficientnet' in exp_name:
                results['model_type'] = 'EfficientNet'
            elif 'mobilenet' in exp_name:
                results['model_type'] = 'MobileNet'
            elif 'densenet' in exp_name:
                results['model_type'] = 'DenseNet'
            elif 'vit' in exp_name:
                results['model_type'] = 'ViT'
            else:
                results['model_type'] = 'Unknown'

            all_results.append(results)

            if results['status'] == 'completed':
                print(f"  ✅ {exp['name']}: {results.get('accuracy', 'N/A')}% accuracy")
            else:
                print(f"  ⏳ {exp['name']}: In progress or incomplete")

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Filter to completed experiments only unless specified
    if not args.include_running:
        results_df = results_df[results_df['status'] == 'completed']

    print(f"\n📈 Found {len(results_df)} completed experiments for analysis")

    # Load crop metadata
    print("\n📂 Loading crop generation metadata...")
    crop_metadata = load_crop_metadata()
    crop_analysis = analyze_crop_quality(crop_metadata)

    for source, analysis in crop_analysis.items():
        print(f"  📊 {source}: {analysis['total_crops']:,} crops generated")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate plots
    if len(results_df) > 0:
        print(f"\n📊 Creating comparison plots...")
        create_comparison_plots(results_df, crop_analysis, output_path)
        print(f"  ✅ Plots saved to {output_path}")

    # Generate report
    print(f"\n📝 Generating comparison report...")
    generate_comparison_report(results_df, crop_analysis, output_path)

    # Save raw data
    results_df.to_csv(output_path / 'experiment_results.csv', index=False)

    with open(output_path / 'crop_analysis.json', 'w') as f:
        json.dump(crop_analysis, f, indent=2, default=str)

    print(f"\n🎉 Comparison analysis completed!")
    print(f"📂 Results saved to: {output_path}")
    print(f"📊 Summary:")

    if len(results_df) > 0:
        print(f"   - {len(results_df)} experiments analyzed")
        if 'accuracy' in results_df.columns:
            best_exp = results_df.loc[results_df['accuracy'].idxmax()]
            print(f"   - Best performance: {best_exp['accuracy']:.2f}% ({best_exp['experiment_name']})")

            # Show performance by source
            source_performance = results_df.groupby('source')['accuracy'].mean().sort_values(ascending=False)
            print(f"   - Performance by source:")
            for source, avg_acc in source_performance.items():
                count = len(results_df[results_df['source'] == source])
                print(f"     * {source}: {avg_acc:.2f}% average ({count} experiments)")

    total_crops = sum(analysis['total_crops'] for analysis in crop_analysis.values())
    print(f"   - Total crops analyzed: {total_crops:,}")

if __name__ == "__main__":
    main()