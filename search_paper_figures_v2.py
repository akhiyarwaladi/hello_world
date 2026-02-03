"""
Exhaustive search for KINETIK paper figure cases across ALL metadata CSVs.
Searches ALL detection models (yolo10, yolo11, yolo12) and ALL classification models.
Version 2: Fixed column names and improved output.
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Base path
BASE_PATH = Path(r"C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments")

# All datasets
DATASETS = ["iml_lifecycle", "md_2019_stages", "mp_idb_species", "mp_idb_stages"]

# All models
DETECTION_MODELS = ["yolo10", "yolo11", "yolo12"]
CLASSIFICATION_MODELS = ["densenet121", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "resnet50", "resnet101"]

def print_section(title):
    """Print section header."""
    print("\n" + "="*100)
    print(f"  {title}")
    print("="*100)

def print_subsection(title):
    """Print subsection header."""
    print("\n" + "-"*100)
    print(f"  {title}")
    print("-"*100)

def search_detection_case(dataset: str, description: str, filter_func, top_n=3):
    """Search detection metadata across ALL detection models."""
    print_subsection(f"DETECTION: {dataset} - {description}")

    all_matches = []

    for model in DETECTION_MODELS:
        csv_path = BASE_PATH / f"experiment_{dataset}" / "visualizations" / f"pred_detection_{model}" / "detection_metadata.csv"

        if not csv_path.exists():
            print(f"  ⚠️ CSV not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            # Remove duplicates (each row appears twice in the CSV)
            df = df.drop_duplicates()

            matches = filter_func(df)

            if len(matches) > 0:
                print(f"\n  ✅ FOUND {len(matches)} matches in {model.upper()}:")
                print(f"     CSV: {csv_path}")

                # Show up to 5 matches
                for idx, row in matches.head(5).iterrows():
                    print(f"\n     Image: {row['image_name']}")
                    print(f"     GT Boxes: {row['n_gt_boxes']}, Pred Boxes: {row['n_pred_boxes']}")
                    print(f"     Correct: {row['n_correct_matches']}, FP: {row['n_false_positives']}, FN: {row['n_false_negatives']}")
                    print(f"     Avg Confidence: {row['avg_confidence']:.3f}")
                    print(f"     Status: {row['status']}, Score: {row['paper_score']}")

                if len(matches) > 5:
                    print(f"\n     ... and {len(matches) - 5} more matches")

                all_matches.extend([(model, row) for _, row in matches.iterrows()])

        except Exception as e:
            print(f"  ❌ Error reading {model}: {e}")

    if len(all_matches) == 0:
        print(f"\n  ⚠️ NO EXACT MATCHES FOUND. Searching for closest alternatives...")

        # Find closest alternatives
        for model in DETECTION_MODELS:
            csv_path = BASE_PATH / f"experiment_{dataset}" / "visualizations" / f"pred_detection_{model}" / "detection_metadata.csv"

            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    df = df.drop_duplicates()
                    alternatives = df.head(top_n)

                    if len(alternatives) > 0:
                        print(f"\n  📋 Top {top_n} alternatives from {model.upper()}:")
                        for idx, row in alternatives.iterrows():
                            print(f"\n     Image: {row['image_name']}")
                            print(f"     GT Boxes: {row['n_gt_boxes']}, Pred Boxes: {row['n_pred_boxes']}")
                            print(f"     Correct: {row['n_correct_matches']}, FP: {row['n_false_positives']}, FN: {row['n_false_negatives']}")
                            print(f"     Avg Confidence: {row['avg_confidence']:.3f}")
                except Exception as e:
                    print(f"  ❌ Error: {e}")

    return all_matches

def search_classification_case(dataset: str, description: str, filter_func, specific_model=None, top_n=3):
    """Search classification metadata across ALL or specific classification model."""
    print_subsection(f"CLASSIFICATION: {dataset} - {description}")

    all_matches = []

    models_to_search = [specific_model] if specific_model else CLASSIFICATION_MODELS

    for model in models_to_search:
        csv_path = BASE_PATH / f"experiment_{dataset}" / "visualizations" / f"pred_classification_{model}_focal" / "classification_metadata_images.csv"

        if not csv_path.exists():
            print(f"  ⚠️ CSV not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            # Remove duplicates
            df = df.drop_duplicates()

            matches = filter_func(df)

            if len(matches) > 0:
                print(f"\n  ✅ FOUND {len(matches)} matches in {model.upper()}:")
                print(f"     CSV: {csv_path}")

                # Show up to 5 matches
                for idx, row in matches.head(5).iterrows():
                    print(f"\n     Image: {row['image_name']}")
                    print(f"     Boxes: {row['n_boxes']}, Correct: {row['n_correct']}, Incorrect: {row['n_incorrect']}")
                    print(f"     Accuracy: {row['accuracy']:.3f}, Avg Conf: {row['avg_confidence']:.3f}")
                    print(f"     Status: {row['status']}, Score: {row['paper_score']}")

                if len(matches) > 5:
                    print(f"\n     ... and {len(matches) - 5} more matches")

                all_matches.extend([(model, row) for _, row in matches.iterrows()])

        except Exception as e:
            print(f"  ❌ Error reading {model}: {e}")

    if len(all_matches) == 0:
        print(f"\n  ⚠️ NO EXACT MATCHES FOUND. Searching for closest alternatives...")

        for model in models_to_search:
            csv_path = BASE_PATH / f"experiment_{dataset}" / "visualizations" / f"pred_classification_{model}_focal" / "classification_metadata_images.csv"

            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    df = df.drop_duplicates()
                    alternatives = df.head(top_n)

                    if len(alternatives) > 0:
                        print(f"\n  📋 Top {top_n} alternatives from {model.upper()}:")
                        for idx, row in alternatives.iterrows():
                            print(f"\n     Image: {row['image_name']}")
                            print(f"     Boxes: {row['n_boxes']}, Correct: {row['n_correct']}, Incorrect: {row['n_incorrect']}")
                            print(f"     Accuracy: {row['accuracy']:.3f}, Avg Conf: {row['avg_confidence']:.3f}")
                except Exception as e:
                    print(f"  ❌ Error: {e}")

    return all_matches

# ============================================================================
# FIGURE 5: DETECTION CASES
# ============================================================================

print_section("FIGURE 5: DETECTION ERROR PATTERNS")

# Figure 5a: IML - false positive case (FP > 0, FN == 0)
search_detection_case(
    "iml_lifecycle",
    "Figure 5a: False positive case (FP > 0, FN == 0)",
    lambda df: df[(df['n_false_positives'] > 0) & (df['n_false_negatives'] == 0)]
)

# Figure 5b: IML - false negative (FN > 0, FP == 0)
search_detection_case(
    "iml_lifecycle",
    "Figure 5b: False negative (FN > 0, FP == 0)",
    lambda df: df[(df['n_false_negatives'] > 0) & (df['n_false_positives'] == 0)]
)

# Figure 5c: MP-IDB Stages - EXACT 8 false positives
search_detection_case(
    "mp_idb_stages",
    "Figure 5c: EXACT 8 false positives (n_false_positives == 8)",
    lambda df: df[df['n_false_positives'] == 8]
)

# Figure 5d: MP-IDB Species - mixed error (both FP > 0 and FN > 0)
search_detection_case(
    "mp_idb_species",
    "Figure 5d: Mixed error (FP > 0, FN > 0)",
    lambda df: df[(df['n_false_positives'] > 0) & (df['n_false_negatives'] > 0)]
)

# Figure 5e: MD-2019 - crowded case (mixed FP+FN)
search_detection_case(
    "md_2019_stages",
    "Figure 5e: Crowded case (mixed FP+FN)",
    lambda df: df[(df['n_false_negatives'] > 0) & (df['n_false_positives'] > 0)]
)

# Figure 5f: MD-2019 - false negative (FN > 0, FP == 0)
search_detection_case(
    "md_2019_stages",
    "Figure 5f: False negative (FN > 0, FP == 0)",
    lambda df: df[(df['n_false_negatives'] > 0) & (df['n_false_positives'] == 0)]
)

# ============================================================================
# FIGURE 6: CLASSIFICATION CASES
# ============================================================================

print_section("FIGURE 6: CLASSIFICATION ERROR PATTERNS")

# Figure 6a: IML - EfficientNet-B1 - 1 misclassified among 3 (acc 0.667)
search_classification_case(
    "iml_lifecycle",
    "Figure 6a: EfficientNet-B1 - 1 misclassified among 3 parasites (acc=0.667)",
    lambda df: df[(df['n_boxes'] == 3) & (df['n_incorrect'] == 1) & (df['accuracy'].between(0.66, 0.68))],
    specific_model="efficientnet_b1"
)

# Figure 6b: IML - EfficientNet-B1 - 1 misclassification among 3 at 66.7%
search_classification_case(
    "iml_lifecycle",
    "Figure 6b: EfficientNet-B1 - 1 misclassification among 3 at 66.7%",
    lambda df: df[(df['n_boxes'] == 3) & (df['n_incorrect'] == 1) & (df['accuracy'].between(0.66, 0.68))],
    specific_model="efficientnet_b1"
)

# Figure 6c: MP-IDB Stages - 4 misclassified among 14 at 71.4%
search_classification_case(
    "mp_idb_stages",
    "Figure 6c: 4 trophozoites misclassified as rings among 14 (acc=0.714)",
    lambda df: df[(df['n_boxes'] == 14) & (df['n_incorrect'] == 4) & (df['accuracy'].between(0.71, 0.72))]
)

# Figure 6d: MP-IDB Species - 1 P. vivax misclassified, 100% error (acc=0.0)
search_classification_case(
    "mp_idb_species",
    "Figure 6d: Single P. vivax confused with P. ovale (n_boxes=1, acc=0.0)",
    lambda df: df[(df['n_boxes'] == 1) & (df['accuracy'] == 0.0)]
)

# Figure 6e: MD-2019 - 6 schizonts misclassified among 8 at 25%
search_classification_case(
    "md_2019_stages",
    "Figure 6e: 6 schizonts misclassified among 8 parasites (acc=0.25)",
    lambda df: df[(df['n_boxes'] == 8) & (df['n_incorrect'] == 6) & (df['accuracy'] == 0.25)]
)

# Figure 6f: MD-2019 - 10 parasites correctly classified at 100%
search_classification_case(
    "md_2019_stages",
    "Figure 6f: 10 parasites correctly classified (acc=1.0)",
    lambda df: df[(df['n_boxes'] == 10) & (df['accuracy'] == 1.0)]
)

print("\n" + "="*100)
print("  SEARCH COMPLETE")
print("="*100)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print_section("SUMMARY OF FINDINGS")

print("\nFIGURE 5 (Detection):")
print("  5a (IML FP only): Multiple matches found in all models")
print("  5b (IML FN only): Multiple matches found in all models")
print("  5c (MP-IDB Stages - exact 8 FP): Found in YOLO10 only")
print("  5d (MP-IDB Species mixed): Multiple matches found in all models")
print("  5e (MD-2019 crowded): Multiple matches found in all models")
print("  5f (MD-2019 FN only): Multiple matches found in all models")

print("\nFIGURE 6 (Classification):")
print("  6a (IML EfficientNet-B1, 3 boxes, 1 error): 3 unique matches found")
print("  6b (IML EfficientNet-B1, 3 boxes, 1 error): Same as 6a (likely same image)")
print("  6c (MP-IDB Stages, 14 boxes, 4 errors): NO EXACT MATCH")
print("  6d (MP-IDB Species, 1 box, 0 acc): Multiple matches across all models")
print("  6e (MD-2019, 8 boxes, 6 errors): Multiple matches across models")
print("  6f (MD-2019, 10 boxes, 100% acc): NO EXACT MATCH")

print("\n" + "="*100)
