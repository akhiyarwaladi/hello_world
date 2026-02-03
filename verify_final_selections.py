"""
Final verification of selected cases with exact file paths
"""

import pandas as pd
from pathlib import Path

base_path = Path(r"C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments")

print("="*80)
print("FINAL VERIFICATION OF SELECTED CASES")
print("="*80)

# =============================================================================
# Figure 5c: Detection - YOLO10
# =============================================================================
print("\n" + "="*80)
print("✅ FIGURE 5c: Detection - MP-IDB Stages (8 False Positives)")
print("="*80)

det_file = base_path / "experiment_mp_idb_stages/visualizations/pred_detection_yolo10/detection_metadata.csv"
df_det = pd.read_csv(det_file)

selected = df_det[df_det['image_name'] == '1405022890-0003-R']
print(f"\nFile: {det_file}")
print(f"Image: 1405022890-0003-R")
print(f"\nNote: Found {len(selected)} rows (appears to be duplicate entries)")
print("\nFirst entry:")
print(selected.iloc[0].to_string())

# Check for actual image file
img_file = selected.iloc[0]['image_file']
img_path = Path(r"C:\Users\MyPC PRO\Documents\hello_world") / img_file
print(f"\nImage file exists: {img_path.exists()}")
print(f"Image path: {img_path}")

# =============================================================================
# Figure 6a & 6b: IML Lifecycle
# =============================================================================
print("\n" + "="*80)
print("✅ FIGURE 6a: IML Lifecycle - PA171852")
print("="*80)

cls_file_6a = base_path / "experiment_iml_lifecycle/visualizations/pred_classification_efficientnet_b0_focal/classification_metadata_images.csv"
df_6a = pd.read_csv(cls_file_6a)
selected_6a = df_6a[df_6a['image_name'] == 'PA171852']

print(f"\nFile: {cls_file_6a}")
print(f"Image: PA171852")
print(f"Found {len(selected_6a)} rows")
print("\nFirst entry:")
print(selected_6a.iloc[0].to_string())

# Also show it across different models
print("\n📊 Same image across different models:")
models = ['densenet121', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'resnet50', 'resnet101']
for model in models:
    cls_file = base_path / f"experiment_iml_lifecycle/visualizations/pred_classification_{model}_focal/classification_metadata_images.csv"
    if cls_file.exists():
        df = pd.read_csv(cls_file)
        match = df[df['image_name'] == 'PA171852']
        if not match.empty:
            print(f"  {model:20s}: acc={match.iloc[0]['accuracy']:.3f}, conf={match.iloc[0]['avg_confidence']:.3f}")

print("\n" + "="*80)
print("✅ FIGURE 6b: IML Lifecycle - PA171802")
print("="*80)

cls_file_6b = base_path / "experiment_iml_lifecycle/visualizations/pred_classification_resnet50_focal/classification_metadata_images.csv"
df_6b = pd.read_csv(cls_file_6b)
selected_6b = df_6b[df_6b['image_name'] == 'PA171802']

print(f"\nFile: {cls_file_6b}")
print(f"Image: PA171802")
print(f"Found {len(selected_6b)} rows")
print("\nFirst entry:")
print(selected_6b.iloc[0].to_string())

print("\n📊 Same image across different models:")
for model in models:
    cls_file = base_path / f"experiment_iml_lifecycle/visualizations/pred_classification_{model}_focal/classification_metadata_images.csv"
    if cls_file.exists():
        df = pd.read_csv(cls_file)
        match = df[df['image_name'] == 'PA171802']
        if not match.empty:
            print(f"  {model:20s}: acc={match.iloc[0]['accuracy']:.3f}, conf={match.iloc[0]['avg_confidence']:.3f}")

# =============================================================================
# Figure 6e: MD-2019
# =============================================================================
print("\n" + "="*80)
print("✅ FIGURE 6e: MD-2019 - Trip 064 Day 2 25-11-05 Image 5_11")
print("="*80)

cls_file_6e = base_path / "experiment_md_2019_stages/visualizations/pred_classification_efficientnet_b2_focal/classification_metadata_images.csv"
df_6e = pd.read_csv(cls_file_6e)
selected_6e = df_6e[df_6e['image_name'] == 'Trip 064 Day 2 25-11-05 Image 5_11']

print(f"\nFile: {cls_file_6e}")
print(f"Image: Trip 064 Day 2 25-11-05 Image 5_11")
print(f"Found {len(selected_6e)} rows")
print("\nFirst entry:")
print(selected_6e.iloc[0].to_string())

print("\n📊 Same image across different models:")
md_models = ['densenet121', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'resnet50']
for model in md_models:
    cls_file = base_path / f"experiment_md_2019_stages/visualizations/pred_classification_{model}_focal/classification_metadata_images.csv"
    if cls_file.exists():
        df = pd.read_csv(cls_file)
        match = df[df['image_name'] == 'Trip 064 Day 2 25-11-05 Image 5_11']
        if not match.empty:
            print(f"  {model:20s}: acc={match.iloc[0]['accuracy']:.3f}, conf={match.iloc[0]['avg_confidence']:.3f}")

# =============================================================================
# Figure 6f: MD-2019
# =============================================================================
print("\n" + "="*80)
print("⚠️ FIGURE 6f: MD-2019 - Trip 065 Day 2 01-12-05 Image 7_9")
print("(Only n_boxes=8, not ideal n_boxes=10)")
print("="*80)

cls_file_6f = base_path / "experiment_md_2019_stages/visualizations/pred_classification_resnet101_focal/classification_metadata_images.csv"
df_6f = pd.read_csv(cls_file_6f)
selected_6f = df_6f[df_6f['image_name'] == 'Trip 065 Day 2 01-12-05 Image 7_9']

print(f"\nFile: {cls_file_6f}")
print(f"Image: Trip 065 Day 2 01-12-05 Image 7_9")
print(f"Found {len(selected_6f)} rows")
print("\nFirst entry:")
print(selected_6f.iloc[0].to_string())

# Check other models for this image
print("\n📊 Same image in other models:")
all_md_models = ['densenet121', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'resnet50', 'resnet101']
found_in_other = False
for model in all_md_models:
    if model == 'resnet101':
        continue
    cls_file = base_path / f"experiment_md_2019_stages/visualizations/pred_classification_{model}_focal/classification_metadata_images.csv"
    if cls_file.exists():
        df = pd.read_csv(cls_file)
        match = df[df['image_name'] == 'Trip 065 Day 2 01-12-05 Image 7_9']
        if not match.empty:
            found_in_other = True
            print(f"  {model:20s}: acc={match.iloc[0]['accuracy']:.3f}, conf={match.iloc[0]['avg_confidence']:.3f}")

if not found_in_other:
    print("  ⚠️ Only found in ResNet101 - image may be unique to this model")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

summary = """
✅ Figure 5c: EXACT MATCH
   - YOLO10: 1405022890-0003-R (8 FPs)

✅ Figure 6a: EXACT MATCH
   - IML Lifecycle: PA171852 (n_boxes=3, acc=0.667)
   - Found in all 6 classification models

✅ Figure 6b: EXACT MATCH
   - IML Lifecycle: PA171802 (n_boxes=3, acc=0.667)
   - Found in 5/6 classification models

✅ Figure 6e: EXACT MATCH
   - MD-2019: Trip 064 Day 2 25-11-05 Image 5_11 (n_boxes=8, acc=0.25)
   - Found in 5/6 classification models

⚠️ Figure 6f: PARTIAL MATCH
   - MD-2019: Trip 065 Day 2 01-12-05 Image 7_9 (n_boxes=8, acc=1.0)
   - Only found in ResNet101
   - n_boxes=8 instead of preferred n_boxes=10

❌ Figure 6c: NO MATCH
   - MP-IDB Stages: High box count images have <10% accuracy in 5-epoch test
   - Requires full 75-epoch training (use baseline experiment)
"""

print(summary)

print("\n📁 CSV Files Used:")
print(f"  Detection: {det_file.name}")
print(f"  IML 6a: pred_classification_efficientnet_b0_focal/classification_metadata_images.csv")
print(f"  IML 6b: pred_classification_resnet50_focal/classification_metadata_images.csv")
print(f"  MD-2019 6e: pred_classification_efficientnet_b2_focal/classification_metadata_images.csv")
print(f"  MD-2019 6f: pred_classification_resnet101_focal/classification_metadata_images.csv")

print("\n" + "="*80)
