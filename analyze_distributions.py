"""
Analyze the actual distributions to understand what's available
"""

import pandas as pd
from pathlib import Path

base_path = Path(r"C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments")

print("="*80)
print("DISTRIBUTION ANALYSIS")
print("="*80)

# =============================================================================
# MP-IDB Stages - What box counts and accuracies are available?
# =============================================================================
print("\n" + "="*80)
print("MP-IDB STAGES - Available Images")
print("="*80)

cls_stages_files = list((base_path / "experiment_mp_idb_stages/visualizations").glob("pred_classification_*/classification_metadata_images.csv"))

# Use first model to see what images exist
first_file = cls_stages_files[0]
df = pd.read_csv(first_file)
model_name = first_file.parent.name.replace('pred_classification_', '')

print(f"\nModel: {model_name.upper()}")
print(f"Total unique images: {df['image_name'].nunique()}")

# Box count distribution
print("\n📊 Box count distribution:")
box_dist = df.groupby('n_boxes').size().sort_index()
print(box_dist)

# Show statistics for different box count ranges
print("\n📈 Statistics by box count ranges:")
for min_boxes in [10, 12, 14, 16]:
    subset = df[df['n_boxes'] >= min_boxes]
    if not subset.empty:
        print(f"\nn_boxes ≥ {min_boxes}: {len(subset)} images")
        print(f"  Accuracy range: {subset['accuracy'].min():.3f} - {subset['accuracy'].max():.3f}")
        print(f"  n_incorrect range: {subset['n_incorrect'].min()} - {subset['n_incorrect'].max()}")

# Find images closest to our target
print("\n🎯 Images closest to target (accuracy 0.70-0.75, n_boxes≥10):")
target_range = df[(df['accuracy'] >= 0.65) & (df['accuracy'] <= 0.80) & (df['n_boxes'] >= 10)]
if not target_range.empty:
    print(target_range[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].head(20).to_string(index=False))

# Show some high box count images
print("\n📦 Highest box count images:")
high_boxes = df.nlargest(15, 'n_boxes')
print(high_boxes[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

# =============================================================================
# MD-2019 - Check for higher box counts with perfect accuracy
# =============================================================================
print("\n" + "="*80)
print("MD-2019 STAGES - Perfect Accuracy Analysis")
print("="*80)

cls_md_files = list((base_path / "experiment_md_2019_stages/visualizations").glob("pred_classification_*/classification_metadata_images.csv"))

first_md = cls_md_files[0]
df_md = pd.read_csv(first_md)
model_md = first_md.parent.name.replace('pred_classification_', '')

print(f"\nModel: {model_md.upper()}")
print(f"Total unique images: {df_md['image_name'].nunique()}")

# Box count distribution
print("\n📊 Box count distribution:")
box_dist_md = df_md.groupby('n_boxes').size().sort_index()
print(box_dist_md)

# Perfect accuracy images
perfect = df_md[df_md['accuracy'] == 1.0]
print(f"\n✨ Perfect accuracy images: {len(perfect)}")
print("\nPerfect accuracy by box count:")
print(perfect.groupby('n_boxes').size().sort_index())

# High accuracy with high box count
print("\n📈 High accuracy (≥0.9) with high box count (≥8):")
high_qual = df_md[(df_md['accuracy'] >= 0.9) & (df_md['n_boxes'] >= 8)]
if not high_qual.empty:
    best = high_qual.nlargest(20, 'n_boxes')
    print(best[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

# Check all models for n_boxes=10 perfect
print("\n🔍 Checking ALL models for n_boxes=10 with perfect accuracy:")
for cls_file in cls_md_files:
    df_check = pd.read_csv(cls_file)
    model_check = cls_file.parent.name.replace('pred_classification_', '')

    exact_10_perfect = df_check[(df_check['n_boxes'] == 10) & (df_check['accuracy'] == 1.0)]
    if not exact_10_perfect.empty:
        print(f"\n✅ {model_check.upper()} - Found {len(exact_10_perfect)} matches:")
        print(exact_10_perfect[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

# Alternative: n_boxes≥9 with perfect accuracy
print("\n🎯 Alternative: n_boxes≥9 with perfect accuracy (all models):")
all_high_perfect = []
for cls_file in cls_md_files:
    df_check = pd.read_csv(cls_file)
    model_check = cls_file.parent.name.replace('pred_classification_', '')

    matches = df_check[(df_check['n_boxes'] >= 9) & (df_check['accuracy'] == 1.0)].copy()
    if not matches.empty:
        matches['model'] = model_check
        all_high_perfect.append(matches)

if all_high_perfect:
    combined = pd.concat(all_high_perfect, ignore_index=True)
    best = combined.nlargest(20, 'n_boxes')
    print(best[['model', 'image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
