"""
Extract exact file paths for all recommended figures from KINETIK paper.
Generates ready-to-use image paths for paper figure generation.
"""

import pandas as pd
from pathlib import Path
import json

BASE_PATH = Path(r"C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments")

def get_detection_image_path(dataset, model, image_name):
    """Get detection visualization image path."""
    csv_path = BASE_PATH / f"experiment_{dataset}" / "visualizations" / f"pred_detection_{model}" / "detection_metadata.csv"

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path).drop_duplicates()
    match = df[df['image_name'] == image_name]

    if len(match) > 0:
        return match.iloc[0]['image_file']
    return None

def get_classification_image_path(dataset, model, image_name):
    """Get classification visualization image path."""
    csv_path = BASE_PATH / f"experiment_{dataset}" / "visualizations" / f"pred_classification_{model}_focal" / "classification_metadata_images.csv"

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path).drop_duplicates()
    match = df[df['image_name'] == image_name]

    if len(match) > 0:
        return match.iloc[0]['image_file']
    return None

def print_section(title):
    print("\n" + "="*100)
    print(f"  {title}")
    print("="*100)

# ============================================================================
# RECOMMENDED FIGURES
# ============================================================================

recommended_figures = {
    "Figure 5a": {
        "type": "detection",
        "dataset": "iml_lifecycle",
        "model": "yolo10",
        "image_name": "PA171698",
        "description": "IML False Positive (GT: 3, Pred: 4, FP: 1)"
    },
    "Figure 5b": {
        "type": "detection",
        "dataset": "iml_lifecycle",
        "model": "yolo10",
        "image_name": "PA171831",
        "description": "IML False Negative (GT: 3, Pred: 2, FN: 1)"
    },
    "Figure 5c": {
        "type": "detection",
        "dataset": "mp_idb_stages",
        "model": "yolo10",
        "image_name": "1405022890-0003-R",
        "description": "MP-IDB Stages - EXACT 8 FP (GT: 24, Pred: 29, FP: 8, FN: 3)"
    },
    "Figure 5d": {
        "type": "detection",
        "dataset": "mp_idb_species",
        "model": "yolo10",
        "image_name": "1704282807-0015-R",
        "description": "MP-IDB Species Mixed (GT: 21, Pred: 21, FP: 3, FN: 3)"
    },
    "Figure 5e": {
        "type": "detection",
        "dataset": "md_2019_stages",
        "model": "yolo10",
        "image_name": "Trip 073 Day 2 01-12-05 Image 1_6",
        "description": "MD-2019 Crowded Mixed (GT: 10, Pred: 11, FP: 3, FN: 2)"
    },
    "Figure 5f": {
        "type": "detection",
        "dataset": "md_2019_stages",
        "model": "yolo11",
        "image_name": "Trip 073 Day 2 01-12-05 Image 1_10",
        "description": "MD-2019 False Negative (GT: 14, Pred: 10, FN: 4)"
    },
    "Figure 6a": {
        "type": "classification",
        "dataset": "iml_lifecycle",
        "model": "efficientnet_b1",
        "image_name": "PA171852",
        "description": "IML EfficientNet-B1 (3 boxes, 1 error, 66.7%)"
    },
    "Figure 6b": {
        "type": "classification",
        "dataset": "iml_lifecycle",
        "model": "efficientnet_b1",
        "image_name": "PA171771",
        "description": "IML EfficientNet-B1 (3 boxes, 1 error, 66.7%)"
    },
    "Figure 6c": {
        "type": "classification",
        "dataset": "mp_idb_stages",
        "model": "N/A",
        "image_name": "N/A",
        "description": "⚠️ NO EXACT MATCH - Manual selection needed"
    },
    "Figure 6d": {
        "type": "classification",
        "dataset": "mp_idb_species",
        "model": "densenet121",
        "image_name": "1305121398-0003-R",
        "description": "MP-IDB Species (1 box, 100% error, cross-model)"
    },
    "Figure 6e": {
        "type": "classification",
        "dataset": "md_2019_stages",
        "model": "densenet121",
        "image_name": "Trip 064 Day 2 25-11-05 Image 5_11",
        "description": "MD-2019 (8 boxes, 6 errors, 25%, cross-model)"
    },
    "Figure 6f": {
        "type": "classification",
        "dataset": "md_2019_stages",
        "model": "resnet101",
        "image_name": "Trip 804 Day 1 02-12-05 Image 3_1",
        "description": "MD-2019 Perfect (6 boxes, 100%, cross-model) - UPDATED from 10 boxes"
    }
}

# ============================================================================
# EXTRACT PATHS
# ============================================================================

print_section("RECOMMENDED FIGURE PATHS - KINETIK PAPER")

all_paths = {}
verified_count = 0
missing_count = 0

for fig_id, fig_info in recommended_figures.items():
    print(f"\n{fig_id}: {fig_info['description']}")

    if fig_info['image_name'] == "N/A":
        print(f"  ⚠️ STATUS: Manual selection required")
        missing_count += 1
        all_paths[fig_id] = {
            "status": "manual_selection_needed",
            "path": None,
            "info": fig_info
        }
        continue

    if fig_info['type'] == 'detection':
        path = get_detection_image_path(fig_info['dataset'], fig_info['model'], fig_info['image_name'])
    else:
        path = get_classification_image_path(fig_info['dataset'], fig_info['model'], fig_info['image_name'])

    if path:
        print(f"  ✅ VERIFIED: {path}")
        verified_count += 1
        all_paths[fig_id] = {
            "status": "verified",
            "path": str(path),
            "info": fig_info
        }
    else:
        print(f"  ❌ NOT FOUND: Check dataset={fig_info['dataset']}, model={fig_info['model']}, image={fig_info['image_name']}")
        missing_count += 1
        all_paths[fig_id] = {
            "status": "not_found",
            "path": None,
            "info": fig_info
        }

# ============================================================================
# SUMMARY
# ============================================================================

print_section("SUMMARY")

print(f"\n  Total Figures: {len(recommended_figures)}")
print(f"  ✅ Verified: {verified_count}")
print(f"  ⚠️ Manual Selection Needed: {missing_count}")

# ============================================================================
# EXPORT TO JSON
# ============================================================================

output_file = Path("recommended_figure_paths.json")
with open(output_file, 'w') as f:
    json.dump(all_paths, f, indent=2)

print(f"\n  📄 Full paths exported to: {output_file.absolute()}")

# ============================================================================
# ALTERNATIVE PATHS (Cross-Model Options)
# ============================================================================

print_section("ALTERNATIVE PATHS (Cross-Model Options)")

# Figure 5b alternatives
print("\nFigure 5b (IML FN) - Alternative: YOLO11")
path_5b_alt = get_detection_image_path("iml_lifecycle", "yolo11", "PA171831")
if path_5b_alt:
    print(f"  ✅ YOLO11: {path_5b_alt}")

# Figure 5d alternatives
print("\nFigure 5d (MP-IDB Species Mixed) - Alternatives:")
for model in ["yolo11", "yolo12"]:
    path_5d_alt = get_detection_image_path("mp_idb_species", model, "1704282807-0015-R")
    if path_5d_alt:
        print(f"  ✅ {model.upper()}: {path_5d_alt}")

# Figure 5f alternatives
print("\nFigure 5f (MD-2019 FN) - Alternative: YOLO12")
path_5f_alt = get_detection_image_path("md_2019_stages", "yolo12", "Trip 073 Day 2 01-12-05 Image 1_10")
if path_5f_alt:
    print(f"  ✅ YOLO12: {path_5f_alt}")

# Figure 6d alternatives (all 6 models)
print("\nFigure 6d (MP-IDB Species 1 box error) - All Models:")
for model in ["densenet121", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "resnet50", "resnet101"]:
    path_6d_alt = get_classification_image_path("mp_idb_species", model, "1305121398-0003-R")
    if path_6d_alt:
        print(f"  ✅ {model.upper()}: {path_6d_alt}")

# Figure 6e alternatives (all tested models)
print("\nFigure 6e (MD-2019 8 boxes, 6 errors) - All Models:")
for model in ["densenet121", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "resnet50"]:
    path_6e_alt = get_classification_image_path("md_2019_stages", model, "Trip 064 Day 2 25-11-05 Image 5_11")
    if path_6e_alt:
        print(f"  ✅ {model.upper()}: {path_6e_alt}")

# Figure 6f alternatives (6 boxes, cross-model)
print("\nFigure 6f (MD-2019 Perfect) - 6 boxes Alternative (Cross-Model):")
for model in ["densenet121", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "resnet50", "resnet101"]:
    path_6f_alt = get_classification_image_path("md_2019_stages", model, "Trip 804 Day 1 02-12-05 Image 3_1")
    if path_6f_alt:
        print(f"  ✅ {model.upper()}: {path_6f_alt}")

print("\n" + "="*100)
print("  EXTRACTION COMPLETE")
print("="*100)
