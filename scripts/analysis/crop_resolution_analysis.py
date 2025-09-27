#!/usr/bin/env python3
"""
Analyze crop resolution and quality across datasets
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pandas as pd

def analyze_crop_dimensions(dataset_path, dataset_name, sample_size=100):
    """Analyze dimensions of crops in a dataset"""
    dataset_path = Path(dataset_path)

    print(f"\n{'='*60}")
    print(f"CROP ANALYSIS: {dataset_name}")
    print(f"{'='*60}")

    # Collect crop files from all classes
    all_crops = []
    class_crops = defaultdict(list)

    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            crop_files = list(class_dir.glob("*.jpg"))
            all_crops.extend(crop_files)
            class_crops[class_name] = crop_files
            print(f"  {class_name}: {len(crop_files)} crops")

    print(f"\nTotal crops found: {len(all_crops)}")

    # Sample crops for analysis
    sample_crops = np.random.choice(all_crops, min(sample_size, len(all_crops)), replace=False)

    # Analyze dimensions
    dimensions = []
    file_sizes = []
    aspect_ratios = []

    print(f"\nAnalyzing {len(sample_crops)} sample crops...")

    for crop_path in sample_crops:
        try:
            # Load image
            img = cv2.imread(str(crop_path))
            if img is not None:
                h, w, c = img.shape
                dimensions.append((w, h))
                aspect_ratios.append(w / h)

                # File size
                file_size = crop_path.stat().st_size
                file_sizes.append(file_size)

        except Exception as e:
            print(f"Error processing {crop_path}: {e}")

    if not dimensions:
        print("No valid crops found!")
        return None

    # Analysis results
    dimensions_array = np.array(dimensions)
    widths = dimensions_array[:, 0]
    heights = dimensions_array[:, 1]
    aspect_ratios = np.array(aspect_ratios)
    file_sizes = np.array(file_sizes)

    # Statistics
    stats = {
        'dataset_name': dataset_name,
        'total_crops': len(all_crops),
        'analyzed_crops': len(dimensions),
        'width_stats': {
            'min': int(np.min(widths)),
            'max': int(np.max(widths)),
            'mean': float(np.mean(widths)),
            'std': float(np.std(widths)),
            'median': float(np.median(widths))
        },
        'height_stats': {
            'min': int(np.min(heights)),
            'max': int(np.max(heights)),
            'mean': float(np.mean(heights)),
            'std': float(np.std(heights)),
            'median': float(np.median(heights))
        },
        'aspect_ratio_stats': {
            'min': float(np.min(aspect_ratios)),
            'max': float(np.max(aspect_ratios)),
            'mean': float(np.mean(aspect_ratios)),
            'std': float(np.std(aspect_ratios))
        },
        'file_size_stats': {
            'min_kb': float(np.min(file_sizes)) / 1024,
            'max_kb': float(np.max(file_sizes)) / 1024,
            'mean_kb': float(np.mean(file_sizes)) / 1024,
            'median_kb': float(np.median(file_sizes)) / 1024
        }
    }

    # Print detailed stats
    print(f"\nDIMENSION ANALYSIS:")
    print(f"  Width  - Min: {stats['width_stats']['min']}, Max: {stats['width_stats']['max']}, Mean: {stats['width_stats']['mean']:.1f} ±{stats['width_stats']['std']:.1f}")
    print(f"  Height - Min: {stats['height_stats']['min']}, Max: {stats['height_stats']['max']}, Mean: {stats['height_stats']['mean']:.1f} ±{stats['height_stats']['std']:.1f}")
    print(f"  Aspect Ratio - Min: {stats['aspect_ratio_stats']['min']:.2f}, Max: {stats['aspect_ratio_stats']['max']:.2f}, Mean: {stats['aspect_ratio_stats']['mean']:.2f}")

    print(f"\nFILE SIZE ANALYSIS:")
    print(f"  Min: {stats['file_size_stats']['min_kb']:.1f} KB")
    print(f"  Max: {stats['file_size_stats']['max_kb']:.1f} KB")
    print(f"  Mean: {stats['file_size_stats']['mean_kb']:.1f} KB")
    print(f"  Median: {stats['file_size_stats']['median_kb']:.1f} KB")

    # Most common dimensions
    dimension_counts = Counter(dimensions)
    print(f"\nMOST COMMON DIMENSIONS:")
    for dim, count in dimension_counts.most_common(10):
        percentage = (count / len(dimensions)) * 100
        print(f"  {dim[0]}x{dim[1]}: {count} crops ({percentage:.1f}%)")

    # Resolution categories
    print(f"\nRESOLUTION CATEGORIES:")
    low_res = sum(1 for w, h in dimensions if max(w, h) < 128)
    medium_res = sum(1 for w, h in dimensions if 128 <= max(w, h) < 224)
    high_res = sum(1 for w, h in dimensions if 224 <= max(w, h) < 512)
    very_high_res = sum(1 for w, h in dimensions if max(w, h) >= 512)

    total = len(dimensions)
    print(f"  Low (<128px): {low_res} ({low_res/total*100:.1f}%)")
    print(f"  Medium (128-223px): {medium_res} ({medium_res/total*100:.1f}%)")
    print(f"  High (224-511px): {high_res} ({high_res/total*100:.1f}%)")
    print(f"  Very High (≥512px): {very_high_res} ({very_high_res/total*100:.1f}%)")

    return stats

def compare_with_paper_standards():
    """Compare current crops with typical paper standards"""
    print(f"\n{'='*60}")
    print("COMPARISON WITH PAPER STANDARDS")
    print(f"{'='*60}")

    print("TYPICAL MALARIA DETECTION PAPER STANDARDS:")
    print("  • ImageNet pretrained models: 224x224")
    print("  • High-resolution analysis: 512x512")
    print("  • Medical imaging standards: 256x256 or 512x512")
    print("  • Transfer learning: 224x224 (ResNet, EfficientNet, etc.)")

    print("\nBENEFITS OF HIGHER RESOLUTION:")
    print("  • Better feature extraction from small parasites")
    print("  • More detailed morphological information")
    print("  • Improved transfer learning from pretrained models")
    print("  • Better generalization across different imaging conditions")

    print("\nDRAWBACKS OF CURRENT LOW RESOLUTION:")
    print("  • Loss of fine-grained parasite details")
    print("  • Limited feature diversity")
    print("  • Poor transfer learning compatibility")
    print("  • Difficulty distinguishing similar classes")

def create_sample_visualization(dataset_path, output_dir, num_samples=16):
    """Create visualization of sample crops from each class"""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating sample visualizations...")

    # Get classes
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        crop_files = list(class_dir.glob("*.jpg"))

        if len(crop_files) == 0:
            continue

        # Sample crops
        sample_files = np.random.choice(crop_files, min(num_samples, len(crop_files)), replace=False)

        # Create grid visualization
        cols = 4
        rows = (len(sample_files) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'{class_name} Samples (n={len(crop_files)})', fontsize=16)

        for idx, crop_file in enumerate(sample_files):
            row = idx // cols
            col = idx % cols

            try:
                img = cv2.imread(str(crop_file))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img_rgb.shape[:2]

                    axes[row, col].imshow(img_rgb)
                    axes[row, col].set_title(f'{w}x{h}', fontsize=10)
                    axes[row, col].axis('off')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[row, col].axis('off')

        # Hide empty subplots
        for idx in range(len(sample_files), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'{class_name}_samples.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualization for {class_name}")

def main():
    """Main analysis function"""

    # Dataset paths
    datasets = {
        'species': {
            'path': 'data/ground_truth_crops/species/crops/train',
            'name': 'MP-IDB Species'
        },
        'stages': {
            'path': 'data/ground_truth_crops/stages/crops/train',
            'name': 'MP-IDB Stages'
        },
        'lifecycle': {
            'path': 'data/ground_truth_crops/lifecycle/crops/train',
            'name': 'IML Lifecycle'
        }
    }

    # Analyze each dataset
    all_stats = {}

    for dataset_key, config in datasets.items():
        stats = analyze_crop_dimensions(config['path'], config['name'])
        if stats:
            all_stats[dataset_key] = stats

            # Create sample visualizations
            create_sample_visualization(
                config['path'],
                f'results/crop_analysis/{dataset_key}',
                num_samples=16
            )

    # Paper standards comparison
    compare_with_paper_standards()

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")

    print(f"{'Dataset':<20} {'Mean Width':<12} {'Mean Height':<12} {'Mean Resolution':<15} {'Standard Target':<15}")
    print("-" * 80)

    for dataset_key, stats in all_stats.items():
        mean_w = stats['width_stats']['mean']
        mean_h = stats['height_stats']['mean']
        mean_res = max(mean_w, mean_h)
        config = datasets[dataset_key]

        print(f"{config['name']:<20} {mean_w:<12.0f} {mean_h:<12.0f} {mean_res:<15.0f} {'224x224':<15}")

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    print("1. RESOLUTION UPGRADE:")
    print("   • Regenerate crops at 224x224 for better paper compatibility")
    print("   • Use bicubic interpolation for upscaling existing crops")
    print("   • Consider 512x512 for high-detail analysis")

    print("\n2. CLASS IMBALANCE SOLUTIONS:")
    print("   • Data augmentation specifically for minority classes")
    print("   • Focal loss or cost-sensitive learning")
    print("   • SMOTE or synthetic data generation")
    print("   • Weighted sampling during training")

    print("\n3. MODEL IMPROVEMENTS:")
    print("   • Use pretrained models (ResNet, EfficientNet) at 224x224")
    print("   • Progressive resizing during training")
    print("   • Multi-scale training approach")
    print("   • Ensemble methods")

    # Save detailed analysis
    output_file = 'results/crop_analysis/detailed_analysis.txt'
    Path('results/crop_analysis').mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("CROP RESOLUTION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        for dataset_key, stats in all_stats.items():
            f.write(f"Dataset: {stats['dataset_name']}\n")
            f.write(f"Total crops: {stats['total_crops']}\n")
            f.write(f"Mean dimensions: {stats['width_stats']['mean']:.1f} x {stats['height_stats']['mean']:.1f}\n")
            f.write(f"Resolution range: {stats['width_stats']['min']}-{stats['width_stats']['max']} x {stats['height_stats']['min']}-{stats['height_stats']['max']}\n")
            f.write(f"Mean file size: {stats['file_size_stats']['mean_kb']:.1f} KB\n")
            f.write("\n")

    print(f"\nDetailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()