"""
Comprehensive MD-2019 Dataset Analysis
Analyzes the Plasmodium falciparum MD-2019 dataset for pipeline integration.
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Paths
BASE_DIR = Path("data/raw/md_2019")
IMG_DIR = BASE_DIR / "Giemsa stained images"
GT_DIR = BASE_DIR / "Ground truth images"
EXCEL_PATH = BASE_DIR / "LifeStages.xlsx"

# Output directory
OUTPUT_DIR = Path("results/md2019_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MD-2019 PLASMODIUM FALCIPARUM DATASET ANALYSIS")
print("=" * 80)
print()

# Load annotations
df = pd.read_excel(EXCEL_PATH)
print(f"Total annotations: {len(df)}")
print(f"Total unique images: {df['imageName'].nunique()}")
print(f"Columns: {list(df.columns)}")
print()

# ============================================================================
# 1. CLASS DISTRIBUTION ANALYSIS
# ============================================================================
print("=" * 80)
print("1. CLASS DISTRIBUTION ANALYSIS")
print("=" * 80)

class_dist = df['stage'].value_counts().sort_values(ascending=False)
class_pct = (class_dist / len(df) * 100).round(2)

print("\nClass Distribution:")
print("-" * 60)
print(f"{'Class':<15} {'Count':>8} {'Percentage':>12} {'Bar':>25}")
print("-" * 60)
for stage, count in class_dist.items():
    pct = class_pct[stage]
    bar = '#' * int(pct)
    print(f"{stage:<15} {count:>8} {pct:>11.2f}% {bar}")
print("-" * 60)
print()

# Class imbalance ratio
max_class = class_dist.max()
min_class = class_dist.min()
imbalance_ratio = max_class / min_class
print(f"Class Imbalance Ratio: {imbalance_ratio:.1f}:1 ({max_class} vs {min_class})")
print()

# ============================================================================
# 2. CLASS MAPPING TO EXISTING MP-IDB STAGES
# ============================================================================
print("=" * 80)
print("2. PROPOSED CLASS MAPPING TO 4-CLASS SYSTEM")
print("=" * 80)

# Define class mappings
class_mapping_4class = {
    'R': 'ring',              # Ring
    'LR-ET': 'ring',          # Late Ring - Early Trophozoite
    'MT': 'trophozoite',      # Mid Trophozoite
    'LT': 'trophozoite',      # Late Trophozoite
    'Esch': 'schizont',       # Early Schizont
    'Lsch': 'schizont',       # Late Schizont
    'Seg': 'schizont',        # Segmented schizont
    'Gam': 'gametocyte',      # Gametocyte
    'DEBRIS': 'EXCLUDE',      # Non-parasite
    'WBC': 'EXCLUDE'          # White Blood Cell
}

# Apply mapping
df['mapped_stage'] = df['stage'].map(class_mapping_4class)

# Count mapped classes
mapped_dist = df[df['mapped_stage'] != 'EXCLUDE']['mapped_stage'].value_counts()
print("\nMapped 4-Class Distribution:")
print("-" * 60)
print(f"{'Class':<15} {'Count':>8} {'Percentage':>12} {'Original Classes'}")
print("-" * 60)
for stage, count in mapped_dist.items():
    original = [k for k, v in class_mapping_4class.items() if v == stage]
    pct = (count / len(df[df['mapped_stage'] != 'EXCLUDE']) * 100)
    print(f"{stage:<15} {count:>8} {pct:>11.2f}% {', '.join(original)}")
print("-" * 60)
print()

excluded_count = len(df[df['mapped_stage'] == 'EXCLUDE'])
print(f"Excluded annotations (DEBRIS + WBC): {excluded_count} ({excluded_count/len(df)*100:.2f}%)")
print(f"Usable annotations: {len(df) - excluded_count} ({(len(df)-excluded_count)/len(df)*100:.2f}%)")
print()

# ============================================================================
# 3. COMPARISON WITH MP-IDB DATASETS
# ============================================================================
print("=" * 80)
print("3. COMPARISON WITH EXISTING MP-IDB DATASETS")
print("=" * 80)

# Load MP-IDB stage data for comparison
mpidb_stages_path = Path("data/raw/mp_idb_stages/annotations.csv")
if mpidb_stages_path.exists():
    mpidb_df = pd.read_csv(mpidb_stages_path)
    mpidb_dist = mpidb_df['stage'].value_counts()

    print("\nMP-IDB Stages Distribution:")
    print("-" * 60)
    for stage, count in mpidb_dist.items():
        pct = (count / len(mpidb_df) * 100)
        print(f"{stage:<15} {count:>8} {pct:>11.2f}%")
    print("-" * 60)
    print()

    # Compare distributions
    print("Comparison (MD-2019 vs MP-IDB Stages):")
    print("-" * 60)
    print(f"{'Stage':<15} {'MD-2019':>12} {'MP-IDB':>12} {'Difference':>12}")
    print("-" * 60)

    for stage in ['ring', 'trophozoite', 'schizont', 'gametocyte']:
        md_count = mapped_dist.get(stage, 0)
        mpidb_count = mpidb_dist.get(stage, 0)

        md_pct = (md_count / len(df[df['mapped_stage'] != 'EXCLUDE']) * 100)
        mpidb_pct = (mpidb_count / len(mpidb_df) * 100)
        diff = md_pct - mpidb_pct

        print(f"{stage:<15} {md_pct:>11.2f}% {mpidb_pct:>11.2f}% {diff:>+11.2f}%")
    print("-" * 60)
else:
    print("MP-IDB Stages dataset not found. Skipping comparison.")
print()

# ============================================================================
# 4. BOUNDING BOX EXTRACTION FROM MASKS
# ============================================================================
print("=" * 80)
print("4. BOUNDING BOX EXTRACTION ANALYSIS")
print("=" * 80)

# Sample 50 images to analyze bounding boxes
sample_images = df['imageName'].unique()[:50]
bbox_stats = []

print(f"\nAnalyzing {len(sample_images)} sample images...")

for img_name in sample_images:
    gt_path = GT_DIR / f"{img_name.replace('.png', '_GT.png')}"

    if not gt_path.exists():
        continue

    # Load mask
    mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get annotations for this image
    img_annotations = df[df['imageName'] == img_name]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h > 0 else 0

        # Find matching annotation by proximity to center
        center_x = x + w / 2
        center_y = y + h / 2

        # Find closest annotation
        distances = np.sqrt((img_annotations['center_x'] - center_x)**2 +
                          (img_annotations['center_y'] - center_y)**2)
        if len(distances) > 0:
            min_dist_idx = distances.idxmin()
            stage = img_annotations.loc[min_dist_idx, 'stage']
            mapped_stage = class_mapping_4class.get(stage, 'UNKNOWN')

            bbox_stats.append({
                'stage': stage,
                'mapped_stage': mapped_stage,
                'width': w,
                'height': h,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'distance_to_annotation': distances[min_dist_idx]
            })

bbox_df = pd.DataFrame(bbox_stats)

print(f"\nBounding Box Statistics (n={len(bbox_df)}):")
print("-" * 60)
print(f"{'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("-" * 60)
print(f"{'Width (px)':<20} {bbox_df['width'].mean():>12.1f} {bbox_df['width'].std():>12.1f} {bbox_df['width'].min():>12.0f} {bbox_df['width'].max():>12.0f}")
print(f"{'Height (px)':<20} {bbox_df['height'].mean():>12.1f} {bbox_df['height'].std():>12.1f} {bbox_df['height'].min():>12.0f} {bbox_df['height'].max():>12.0f}")
print(f"{'Area (px^2)':<20} {bbox_df['area'].mean():>12.1f} {bbox_df['area'].std():>12.1f} {bbox_df['area'].min():>12.0f} {bbox_df['area'].max():>12.0f}")
print(f"{'Aspect Ratio':<20} {bbox_df['aspect_ratio'].mean():>12.2f} {bbox_df['aspect_ratio'].std():>12.2f} {bbox_df['aspect_ratio'].min():>12.2f} {bbox_df['aspect_ratio'].max():>12.2f}")
print(f"{'Center Distance':<20} {bbox_df['distance_to_annotation'].mean():>12.1f} {bbox_df['distance_to_annotation'].std():>12.1f} {bbox_df['distance_to_annotation'].min():>12.1f} {bbox_df['distance_to_annotation'].max():>12.1f}")
print("-" * 60)
print()

# Per-class bbox statistics
print("Bounding Box Statistics by Mapped Class:")
print("-" * 80)
for stage in ['ring', 'trophozoite', 'schizont', 'gametocyte']:
    stage_data = bbox_df[bbox_df['mapped_stage'] == stage]
    if len(stage_data) > 0:
        print(f"\n{stage.upper()} (n={len(stage_data)}):")
        print(f"  Width:  {stage_data['width'].mean():6.1f} ± {stage_data['width'].std():5.1f} px")
        print(f"  Height: {stage_data['height'].mean():6.1f} ± {stage_data['height'].std():5.1f} px")
        print(f"  Area:   {stage_data['area'].mean():6.0f} ± {stage_data['area'].std():5.0f} px²")
print()

# ============================================================================
# 5. TRAIN/VAL/TEST SPLIT RECOMMENDATION
# ============================================================================
print("=" * 80)
print("5. TRAIN/VAL/TEST SPLIT RECOMMENDATIONS")
print("=" * 80)

# Exclude DEBRIS and WBC
usable_df = df[df['mapped_stage'] != 'EXCLUDE'].copy()
print(f"\nUsable annotations: {len(usable_df)} (excludes DEBRIS and WBC)")
print(f"Usable unique images: {usable_df['imageName'].nunique()}")
print()

# Analyze Gametocyte issue
gam_count = len(usable_df[usable_df['mapped_stage'] == 'gametocyte'])
print(f"CRITICAL ISSUE: Gametocyte class has only {gam_count} samples!")
print("Recommendation: Consider excluding Gametocyte class OR merging with another dataset")
print()

# Recommend stratified split with oversampling
print("Recommended Split Strategy:")
print("-" * 60)
print("Option 1: Standard 66/17/17 split")
print("  - Train: 66% (~1,982 samples)")
print("  - Val:   17% (~511 samples)")
print("  - Test:  17% (~511 samples)")
print("  - Issues: Gametocyte class will be extremely underrepresented")
print()
print("Option 2: Image-level stratified split (recommended)")
print("  - Split by images (not annotations) to avoid data leakage")
print("  - Use stratified sampling based on class distribution")
print("  - Apply oversampling for minority classes during training")
print()
print("Option 3: Merge with MP-IDB Stages")
print("  - Combine MD-2019 + MP-IDB Stages for larger dataset")
print("  - More balanced class distribution")
print("  - Better generalization across different image sources")
print("-" * 60)
print()

# ============================================================================
# 6. INTEGRATION STRATEGY RECOMMENDATIONS
# ============================================================================
print("=" * 80)
print("6. INTEGRATION STRATEGY RECOMMENDATIONS")
print("=" * 80)

print("\n" + "=" * 80)
print("STRATEGY 1: SEPARATE MD-2019 DATASET")
print("=" * 80)
print("Pros:")
print("  + Clean single-species (P. falciparum) dataset")
print("  + High-quality binary segmentation masks available")
print("  + Larger sample size (3,004 usable annotations)")
print("  + Can benchmark granular stage classification")
print()
print("Cons:")
print("  - Extreme class imbalance (Gametocyte: 2 samples)")
print("  - Requires excluding 20% of data (DEBRIS + WBC)")
print("  - Gametocyte class essentially unusable")
print("  - Different image resolution (1600x1200 vs MP-IDB)")
print()

print("=" * 80)
print("STRATEGY 2: MERGE WITH MP-IDB STAGES (RECOMMENDED)")
print("=" * 80)
print("Pros:")
print("  + Larger combined dataset (~3,100+ samples after exclusion)")
print("  + More balanced Gametocyte representation")
print("  + Better generalization across image sources")
print("  + Maintains 4-class lifecycle classification")
print("  + Can still use binary masks for detection training")
print()
print("Cons:")
print("  - Different image characteristics (resolution, staining)")
print("  - Requires harmonizing annotation formats")
print("  - Need to validate cross-dataset consistency")
print()

print("=" * 80)
print("STRATEGY 3: 10-CLASS GRANULAR CLASSIFICATION (EXPERIMENTAL)")
print("=" * 80)
print("Pros:")
print("  + Fine-grained lifecycle stage classification")
print("  + Can benchmark on granular stages")
print("  + Unique contribution (most datasets use 4 classes)")
print()
print("Cons:")
print("  - Severe class imbalance (Gametocyte: 2, WBC: 51)")
print("  - Mixed parasite/non-parasite classes")
print("  - Requires sophisticated handling of minority classes")
print("  - Difficult to compare with existing work")
print()

# ============================================================================
# 7. SAVE ANALYSIS RESULTS
# ============================================================================
print("=" * 80)
print("7. SAVING ANALYSIS RESULTS")
print("=" * 80)

# Save class distribution
class_dist_df = pd.DataFrame({
    'class': class_dist.index,
    'count': class_dist.values,
    'percentage': class_pct.values
})
class_dist_df.to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'class_distribution.csv'}")

# Save mapped distribution
mapped_dist_df = pd.DataFrame({
    'class': mapped_dist.index,
    'count': mapped_dist.values,
    'percentage': (mapped_dist.values / mapped_dist.sum() * 100).round(2)
})
mapped_dist_df.to_csv(OUTPUT_DIR / "mapped_4class_distribution.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'mapped_4class_distribution.csv'}")

# Save bbox statistics
bbox_df.to_csv(OUTPUT_DIR / "bbox_statistics.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'bbox_statistics.csv'}")

# Save summary JSON
summary = {
    'dataset': 'MD-2019 Plasmodium falciparum',
    'total_annotations': int(len(df)),
    'total_images': int(df['imageName'].nunique()),
    'usable_annotations': int(len(usable_df)),
    'excluded_annotations': int(excluded_count),
    'class_distribution': class_dist.to_dict(),
    'mapped_4class_distribution': mapped_dist.to_dict(),
    'class_imbalance_ratio': float(imbalance_ratio),
    'bbox_statistics': {
        'mean_width': float(bbox_df['width'].mean()),
        'mean_height': float(bbox_df['height'].mean()),
        'mean_area': float(bbox_df['area'].mean()),
        'mean_aspect_ratio': float(bbox_df['aspect_ratio'].mean())
    },
    'recommendations': {
        'strategy': 'MERGE_WITH_MP_IDB_STAGES',
        'class_mapping': 'Use 4-class system (ring/trophozoite/schizont/gametocyte)',
        'exclude_classes': ['DEBRIS', 'WBC'],
        'critical_issues': ['Gametocyte class has only 2 samples'],
        'split_strategy': 'Image-level stratified split to avoid data leakage'
    }
}

with open(OUTPUT_DIR / "analysis_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved: {OUTPUT_DIR / 'analysis_summary.json'}")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print()

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================
print("Creating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Original class distribution
ax1 = plt.subplot(2, 3, 1)
colors = sns.color_palette("husl", len(class_dist))
bars = ax1.bar(range(len(class_dist)), class_dist.values, color=colors)
ax1.set_xticks(range(len(class_dist)))
ax1.set_xticklabels(class_dist.index, rotation=45, ha='right')
ax1.set_ylabel('Count')
ax1.set_title('Original 10-Class Distribution', fontweight='bold', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, class_dist.values)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             str(val), ha='center', va='bottom', fontsize=9)

# 2. Mapped 4-class distribution
ax2 = plt.subplot(2, 3, 2)
colors_4class = sns.color_palette("Set2", len(mapped_dist))
bars = ax2.bar(range(len(mapped_dist)), mapped_dist.values, color=colors_4class)
ax2.set_xticks(range(len(mapped_dist)))
ax2.set_xticklabels(mapped_dist.index, rotation=45, ha='right')
ax2.set_ylabel('Count')
ax2.set_title('Mapped 4-Class Distribution (Excluding DEBRIS/WBC)', fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, mapped_dist.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Class imbalance visualization
ax3 = plt.subplot(2, 3, 3)
imbalance_data = class_dist.sort_values()
colors_imb = ['red' if x < 100 else 'orange' if x < 300 else 'green' for x in imbalance_data.values]
ax3.barh(range(len(imbalance_data)), imbalance_data.values, color=colors_imb)
ax3.set_yticks(range(len(imbalance_data)))
ax3.set_yticklabels(imbalance_data.index)
ax3.set_xlabel('Count')
ax3.set_title('Class Imbalance (Sorted)', fontweight='bold', fontsize=12)
ax3.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Severe (<100)')
ax3.axvline(x=300, color='orange', linestyle='--', alpha=0.5, label='Moderate (<300)')
ax3.legend(loc='lower right')
ax3.grid(axis='x', alpha=0.3)

# 4. Bounding box size distribution
ax4 = plt.subplot(2, 3, 4)
ax4.hist(bbox_df['width'], bins=30, alpha=0.6, label='Width', color='blue')
ax4.hist(bbox_df['height'], bins=30, alpha=0.6, label='Height', color='red')
ax4.set_xlabel('Size (pixels)')
ax4.set_ylabel('Frequency')
ax4.set_title('Bounding Box Size Distribution', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Bounding box aspect ratio
ax5 = plt.subplot(2, 3, 5)
for stage in ['ring', 'trophozoite', 'schizont']:
    stage_data = bbox_df[bbox_df['mapped_stage'] == stage]
    if len(stage_data) > 0:
        ax5.scatter(stage_data['width'], stage_data['height'],
                   alpha=0.5, label=stage, s=30)
ax5.set_xlabel('Width (pixels)')
ax5.set_ylabel('Height (pixels)')
ax5.set_title('Bounding Box Dimensions by Class', fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(alpha=0.3)

# Add diagonal line (aspect ratio = 1)
max_val = max(bbox_df['width'].max(), bbox_df['height'].max())
ax5.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Aspect ratio = 1')

# 6. Parasites per image distribution
ax6 = plt.subplot(2, 3, 6)
parasites_per_img = df.groupby('imageName').size()
ax6.hist(parasites_per_img, bins=np.arange(1, parasites_per_img.max()+2)-0.5,
         color='teal', alpha=0.7, edgecolor='black')
ax6.set_xlabel('Parasites per Image')
ax6.set_ylabel('Frequency')
ax6.set_title('Distribution of Parasites per Image', fontweight='bold', fontsize=12)
ax6.grid(axis='y', alpha=0.3)

# Add statistics
mean_parasites = parasites_per_img.mean()
median_parasites = parasites_per_img.median()
ax6.axvline(mean_parasites, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_parasites:.2f}')
ax6.axvline(median_parasites, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_parasites:.0f}')
ax6.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comprehensive_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'comprehensive_analysis.png'}")

plt.close()

print("\n✅ All visualizations created successfully!")
print()
print("=" * 80)
print("FINAL RECOMMENDATIONS")
print("=" * 80)
print()
print("RECOMMENDED INTEGRATION STRATEGY: MERGE WITH MP-IDB STAGES")
print()
print("Implementation Steps:")
print("  1. Exclude DEBRIS and WBC classes (742 annotations)")
print("  2. Map granular stages to 4-class system:")
print("     - Ring: R + LR-ET (849 samples)")
print("     - Trophozoite: MT + LT (701 samples)")
print("     - Schizont: Esch + Lsch + Seg (1,369 samples)")
print("     - Gametocyte: Gam (2 samples - CRITICAL)")
print("  3. Extract bounding boxes from binary masks using cv2.findContours()")
print("  4. Convert to YOLO format with normalized coordinates")
print("  5. Merge with MP-IDB Stages dataset for balanced Gametocyte class")
print("  6. Use image-level stratified split (66/17/17) to avoid data leakage")
print("  7. Apply medical-safe augmentation during training")
print()
print("Expected Benefits:")
print("  + Larger training set (~3,000+ samples)")
print("  + Better Gametocyte representation")
print("  + Cross-dataset generalization")
print("  + High-quality binary masks for detection training")
print()
print("Potential Challenges:")
print("  ! Different image resolutions (MD-2019: 1600x1200, MP-IDB: varies)")
print("  ! Different staining/acquisition protocols")
print("  ! Need to validate annotation consistency")
print()
print("=" * 80)
