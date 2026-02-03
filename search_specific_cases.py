"""
Detailed search for specific missing cases
"""

import pandas as pd
from pathlib import Path

base_path = Path(r"C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments")

print("="*80)
print("DETAILED SEARCH FOR SPECIFIC CASES")
print("="*80)

# =============================================================================
# FIGURE 6c: MP-IDB Stages - n_boxes=14, n_incorrect=4, accuracy≈0.714
# =============================================================================
print("\n" + "="*80)
print("FIGURE 6c: MP-IDB Stages - Detailed Search")
print("="*80)

cls_stages_files = list((base_path / "experiment_mp_idb_stages/visualizations").glob("pred_classification_*/classification_metadata_images.csv"))

print("\n1️⃣ EXACT MATCH: n_boxes=14, n_incorrect=4")
for cls_file in cls_stages_files:
    df = pd.read_csv(cls_file)
    model_name = cls_file.parent.name.replace('pred_classification_', '')

    matches = df[(df['n_boxes'] == 14) & (df['n_incorrect'] == 4)]
    if not matches.empty:
        print(f"\n✅ {model_name.upper()}:")
        print(matches[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

print("\n2️⃣ RELAXED: n_boxes=14, n_incorrect=3-5 (accuracy 0.64-0.79)")
all_close = []
for cls_file in cls_stages_files:
    df = pd.read_csv(cls_file)
    model_name = cls_file.parent.name.replace('pred_classification_', '')

    matches = df[(df['n_boxes'] == 14) & (df['n_incorrect'] >= 3) & (df['n_incorrect'] <= 5)].copy()
    if not matches.empty:
        matches['model'] = model_name
        all_close.append(matches)

if all_close:
    combined = pd.concat(all_close, ignore_index=True)
    print(combined[['model', 'image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

print("\n3️⃣ ALTERNATIVE: n_incorrect=4, n_boxes=12-16 (accuracy 0.67-0.75)")
all_alt = []
for cls_file in cls_stages_files:
    df = pd.read_csv(cls_file)
    model_name = cls_file.parent.name.replace('pred_classification_', '')

    matches = df[(df['n_incorrect'] == 4) & (df['n_boxes'] >= 12) & (df['n_boxes'] <= 16)].copy()
    if not matches.empty:
        matches['model'] = model_name
        all_alt.append(matches)

if all_alt:
    combined = pd.concat(all_alt, ignore_index=True)
    print(combined[['model', 'image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

print("\n4️⃣ BROADER: n_boxes≥10, accuracy 0.70-0.75")
all_broad = []
for cls_file in cls_stages_files:
    df = pd.read_csv(cls_file)
    model_name = cls_file.parent.name.replace('pred_classification_', '')

    matches = df[(df['n_boxes'] >= 10) & (df['accuracy'] >= 0.70) & (df['accuracy'] <= 0.75)].copy()
    if not matches.empty:
        matches['model'] = model_name
        all_broad.append(matches)

if all_broad:
    combined = pd.concat(all_broad, ignore_index=True)
    best = combined.nlargest(15, 'n_boxes')
    print(best[['model', 'image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

# =============================================================================
# FIGURE 6f: MD-2019 - Better options with n_boxes=10 or higher
# =============================================================================
print("\n" + "="*80)
print("FIGURE 6f: MD-2019 - Perfect Accuracy Cases (n_boxes≥8)")
print("="*80)

cls_md_files = list((base_path / "experiment_md_2019_stages/visualizations").glob("pred_classification_*/classification_metadata_images.csv"))

all_perfect = []
for cls_file in cls_md_files:
    df = pd.read_csv(cls_file)
    model_name = cls_file.parent.name.replace('pred_classification_', '')

    matches = df[(df['n_boxes'] >= 8) & (df['accuracy'] == 1.0)].copy()
    if not matches.empty:
        matches['model'] = model_name
        all_perfect.append(matches)

if all_perfect:
    combined = pd.concat(all_perfect, ignore_index=True)

    # Show distribution
    print(f"\nTotal perfect accuracy images (n_boxes≥8): {len(combined)}")
    print(f"\nBox count distribution:")
    print(combined['n_boxes'].value_counts().sort_index())

    # Show best options (highest box count)
    best = combined.nlargest(20, 'n_boxes')
    print(f"\n🎯 TOP 20 MATCHES (highest box count):")
    print(best[['model', 'image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

    # Specifically look for n_boxes=10
    exact_10 = combined[combined['n_boxes'] == 10]
    if not exact_10.empty:
        print(f"\n✨ EXACT n_boxes=10 matches:")
        print(exact_10[['model', 'image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

print("\n" + "="*80)
print("SEARCH COMPLETE")
print("="*80)
