"""
Search for specific cases matching KINETIK paper's figure descriptions
across ALL models and datasets in the current experiment.
"""

import pandas as pd
import os
from pathlib import Path

# Base path
base_path = Path(r"C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments")

print("="*80)
print("SEARCHING FOR KINETIK PAPER FIGURE MATCHES")
print("="*80)

# =============================================================================
# FIGURE 5c: Detection - MP-IDB Stages with 8 False Positives (or closest)
# =============================================================================
print("\n" + "="*80)
print("FIGURE 5c: DETECTION - MP-IDB Stages (need 8 FPs, or closest)")
print("="*80)

detection_files = [
    base_path / "experiment_mp_idb_stages/visualizations/pred_detection_yolo10/detection_metadata.csv",
    base_path / "experiment_mp_idb_stages/visualizations/pred_detection_yolo11/detection_metadata.csv",
    base_path / "experiment_mp_idb_stages/visualizations/pred_detection_yolo12/detection_metadata.csv"
]

all_fp_candidates = []

for det_file in detection_files:
    if det_file.exists():
        df = pd.read_csv(det_file)
        model_name = det_file.parent.name.replace('pred_detection_', '')

        # Find images with false positives
        fp_images = df[df['n_false_positives'] > 0].copy()
        fp_images['model'] = model_name
        fp_images['file_path'] = str(det_file)

        all_fp_candidates.append(fp_images)

        # Show exact match if exists
        exact_8fp = df[df['n_false_positives'] == 8]
        if not exact_8fp.empty:
            print(f"\n🎯 EXACT MATCH - {model_name.upper()} - 8 False Positives:")
            print(f"File: {det_file}")
            print(exact_8fp[['image_name', 'n_gt_boxes', 'n_pred_boxes', 'n_correct_matches',
                             'n_false_positives', 'n_false_negatives', 'avg_confidence']].to_string(index=False))

if all_fp_candidates:
    all_fps = pd.concat(all_fp_candidates, ignore_index=True)
    all_fps['fp_diff'] = abs(all_fps['n_false_positives'] - 8)
    closest = all_fps.nsmallest(10, 'fp_diff')

    print("\n📊 CLOSEST MATCHES (Top 10 by FP count):")
    print(closest[['model', 'image_name', 'n_gt_boxes', 'n_pred_boxes', 'n_correct_matches',
                   'n_false_positives', 'n_false_negatives', 'avg_confidence']].to_string(index=False))

# =============================================================================
# FIGURE 6a & 6b: IML Lifecycle - n_boxes=3, n_incorrect=1, accuracy≈0.667
# =============================================================================
print("\n" + "="*80)
print("FIGURE 6a & 6b: CLASSIFICATION - IML Lifecycle")
print("Need: n_boxes=3, n_incorrect=1, accuracy≈0.667")
print("="*80)

cls_iml_files = list((base_path / "experiment_iml_lifecycle/visualizations").glob("pred_classification_*/classification_metadata_images.csv"))

iml_candidates = []

for cls_file in cls_iml_files:
    if cls_file.exists():
        df = pd.read_csv(cls_file)
        model_name = cls_file.parent.name.replace('pred_classification_', '')

        # Find images with n_boxes=3, n_incorrect=1
        matches = df[(df['n_boxes'] == 3) & (df['n_incorrect'] == 1)].copy()
        matches['model'] = model_name
        matches['file_path'] = str(cls_file)

        if not matches.empty:
            iml_candidates.append(matches)
            print(f"\n✅ MATCHES FOUND - {model_name.upper()}:")
            print(f"File: {cls_file}")
            print(matches[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

if not iml_candidates:
    print("\n⚠️ No exact matches. Showing closest alternatives...")
    for cls_file in cls_iml_files:
        df = pd.read_csv(cls_file)
        model_name = cls_file.parent.name.replace('pred_classification_', '')

        # Find images with n_boxes=3 (any accuracy)
        close = df[df['n_boxes'] == 3].copy()
        if not close.empty:
            print(f"\n📊 {model_name.upper()} - Images with n_boxes=3:")
            print(close[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].head(5).to_string(index=False))

# =============================================================================
# FIGURE 6c: MP-IDB Stages - n_boxes=14, n_incorrect=4, accuracy≈0.714
# =============================================================================
print("\n" + "="*80)
print("FIGURE 6c: CLASSIFICATION - MP-IDB Stages")
print("Need: n_boxes=14, n_incorrect=4, accuracy≈0.714")
print("="*80)

cls_stages_files = list((base_path / "experiment_mp_idb_stages/visualizations").glob("pred_classification_*/classification_metadata_images.csv"))

stages_candidates = []

for cls_file in cls_stages_files:
    if cls_file.exists():
        df = pd.read_csv(cls_file)
        model_name = cls_file.parent.name.replace('pred_classification_', '')

        # Find images with n_boxes=14, n_incorrect=4
        matches = df[(df['n_boxes'] == 14) & (df['n_incorrect'] == 4)].copy()
        matches['model'] = model_name
        matches['file_path'] = str(cls_file)

        if not matches.empty:
            stages_candidates.append(matches)
            print(f"\n✅ MATCHES FOUND - {model_name.upper()}:")
            print(f"File: {cls_file}")
            print(matches[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

if not stages_candidates:
    print("\n⚠️ No exact matches. Showing closest alternatives...")
    for cls_file in cls_stages_files:
        df = pd.read_csv(cls_file)
        model_name = cls_file.parent.name.replace('pred_classification_', '')

        # Find images with n_incorrect=4 and high box count
        close = df[(df['n_incorrect'] == 4) & (df['n_boxes'] >= 10)].copy()
        if not close.empty:
            print(f"\n📊 {model_name.upper()} - Images with n_incorrect=4, n_boxes≥10:")
            print(close[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].head(5).to_string(index=False))

# =============================================================================
# FIGURE 6e: MD-2019 - n_boxes=8, n_incorrect=6, accuracy=0.25
# =============================================================================
print("\n" + "="*80)
print("FIGURE 6e: CLASSIFICATION - MD-2019 Stages")
print("Need: n_boxes=8, n_incorrect=6, accuracy=0.25")
print("="*80)

cls_md_files = list((base_path / "experiment_md_2019_stages/visualizations").glob("pred_classification_*/classification_metadata_images.csv"))

md_e_candidates = []

for cls_file in cls_md_files:
    if cls_file.exists():
        df = pd.read_csv(cls_file)
        model_name = cls_file.parent.name.replace('pred_classification_', '')

        # Find images with n_boxes=8, n_incorrect=6
        matches = df[(df['n_boxes'] == 8) & (df['n_incorrect'] == 6)].copy()
        matches['model'] = model_name
        matches['file_path'] = str(cls_file)

        if not matches.empty:
            md_e_candidates.append(matches)
            print(f"\n✅ MATCHES FOUND - {model_name.upper()}:")
            print(f"File: {cls_file}")
            print(matches[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

if not md_e_candidates:
    print("\n⚠️ No exact matches. Showing closest alternatives...")
    for cls_file in cls_md_files:
        df = pd.read_csv(cls_file)
        model_name = cls_file.parent.name.replace('pred_classification_', '')

        # Find images with low accuracy (0.2-0.3) and moderate box count
        close = df[(df['accuracy'] >= 0.20) & (df['accuracy'] <= 0.30) & (df['n_boxes'] >= 6)].copy()
        if not close.empty:
            print(f"\n📊 {model_name.upper()} - Images with accuracy 0.20-0.30, n_boxes≥6:")
            print(close[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].head(5).to_string(index=False))

# =============================================================================
# FIGURE 6f: MD-2019 - n_boxes≥8, accuracy=1.0 (perfect)
# =============================================================================
print("\n" + "="*80)
print("FIGURE 6f: CLASSIFICATION - MD-2019 Stages")
print("Need: n_boxes≥8, accuracy=1.0 (100% correct)")
print("="*80)

md_f_candidates = []

for cls_file in cls_md_files:
    if cls_file.exists():
        df = pd.read_csv(cls_file)
        model_name = cls_file.parent.name.replace('pred_classification_', '')

        # Find images with high box count and perfect accuracy
        matches = df[(df['n_boxes'] >= 8) & (df['accuracy'] == 1.0)].copy()
        matches['model'] = model_name
        matches['file_path'] = str(cls_file)

        if not matches.empty:
            md_f_candidates.append(matches)
            print(f"\n✅ MATCHES FOUND - {model_name.upper()}:")
            print(f"File: {cls_file}")
            print(matches[['image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].head(10).to_string(index=False))

if md_f_candidates:
    all_perfect = pd.concat(md_f_candidates, ignore_index=True)
    # Prefer n_boxes=10, but show all
    best = all_perfect.nlargest(10, 'n_boxes')
    print(f"\n🎯 BEST MATCHES (highest box count):")
    print(best[['model', 'image_name', 'n_boxes', 'n_correct', 'n_incorrect', 'accuracy', 'avg_confidence']].to_string(index=False))

print("\n" + "="*80)
print("SEARCH COMPLETE")
print("="*80)
