"""
Focused search for Figure 6c and 6f alternatives.
Relaxed criteria to find closest matches.
"""

import pandas as pd
from pathlib import Path

BASE_PATH = Path(r"C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251207_233941\experiments")

CLASSIFICATION_MODELS = ["densenet121", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "resnet50", "resnet101"]

def print_section(title):
    print("\n" + "="*100)
    print(f"  {title}")
    print("="*100)

def search_figure_6c_alternatives():
    """
    Figure 6c: MP-IDB Stages - 14 boxes, 4 errors, 71.4% accuracy
    Search with relaxed criteria.
    """
    print_section("FIGURE 6c: MP-IDB Stages - Relaxed Search")

    print("\nOriginal criteria:")
    print("  - n_boxes = 14")
    print("  - n_incorrect = 4")
    print("  - accuracy = 0.714 (71.4%)")

    all_candidates = []

    for model in CLASSIFICATION_MODELS:
        csv_path = BASE_PATH / "experiment_mp_idb_stages" / "visualizations" / f"pred_classification_{model}_focal" / "classification_metadata_images.csv"

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path).drop_duplicates()

        # Search 1: Exact 4 errors with nearby accuracy
        search1 = df[(df['n_incorrect'] == 4) & (df['accuracy'].between(0.70, 0.75))]

        # Search 2: 13-15 boxes with 4 errors
        search2 = df[(df['n_boxes'].between(13, 15)) & (df['n_incorrect'] == 4)]

        # Search 3: Accuracy 0.71-0.72 (71-72%)
        search3 = df[df['accuracy'].between(0.71, 0.72)]

        # Search 4: High box count with similar error ratio (4/14 = 28.6% error)
        search4 = df[
            (df['n_boxes'] >= 10) &
            ((df['n_incorrect'] / df['n_boxes']).between(0.25, 0.32))
        ].copy()
        search4['error_ratio'] = search4['n_incorrect'] / search4['n_boxes']

        if len(search1) > 0:
            print(f"\n  🔍 {model.upper()} - Search 1 (4 errors, 70-75% acc): {len(search1)} matches")
            for idx, row in search1.head(3).iterrows():
                print(f"     {row['image_name']}: {row['n_boxes']} boxes, {row['n_incorrect']} errors, {row['accuracy']:.3f} acc")
                all_candidates.append((model, 'search1', row))

        if len(search2) > 0:
            print(f"\n  🔍 {model.upper()} - Search 2 (13-15 boxes, 4 errors): {len(search2)} matches")
            for idx, row in search2.head(3).iterrows():
                print(f"     {row['image_name']}: {row['n_boxes']} boxes, {row['n_incorrect']} errors, {row['accuracy']:.3f} acc")
                all_candidates.append((model, 'search2', row))

        if len(search3) > 0:
            print(f"\n  🔍 {model.upper()} - Search 3 (71-72% accuracy): {len(search3)} matches")
            for idx, row in search3.head(3).iterrows():
                print(f"     {row['image_name']}: {row['n_boxes']} boxes, {row['n_incorrect']} errors, {row['accuracy']:.3f} acc")
                all_candidates.append((model, 'search3', row))

        if len(search4) > 0:
            search4_sorted = search4.sort_values('n_boxes', ascending=False)
            print(f"\n  🔍 {model.upper()} - Search 4 (25-32% error ratio, 10+ boxes): {len(search4)} matches")
            for idx, row in search4_sorted.head(3).iterrows():
                print(f"     {row['image_name']}: {row['n_boxes']} boxes, {row['n_incorrect']} errors, {row['accuracy']:.3f} acc, {row['error_ratio']:.1%} error")
                all_candidates.append((model, 'search4', row))

    print("\n" + "-"*100)
    print("  RECOMMENDATION FOR FIGURE 6c")
    print("-"*100)

    if len(all_candidates) > 0:
        print("\n  Best alternatives (prioritized):")
        print("  1. Use image with 4 errors and accuracy closest to 71.4%")
        print("  2. Update paper description to match actual data")
        print("  3. Choose highest box count with similar error ratio (28-29%)")
    else:
        print("\n  ⚠️ No close alternatives found. Consider:")
        print("  - Manually inspecting MP-IDB Stages dataset for representative error cases")
        print("  - Using a different error pattern that better represents the dataset")

    return all_candidates

def search_figure_6f_alternatives():
    """
    Figure 6f: MD-2019 - 10 boxes, 100% accuracy
    Search with relaxed criteria.
    """
    print_section("FIGURE 6f: MD-2019 - Relaxed Search")

    print("\nOriginal criteria:")
    print("  - n_boxes = 10")
    print("  - accuracy = 1.0 (100%)")

    all_candidates = []

    for model in CLASSIFICATION_MODELS:
        csv_path = BASE_PATH / "experiment_md_2019_stages" / "visualizations" / f"pred_classification_{model}_focal" / "classification_metadata_images.csv"

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path).drop_duplicates()

        # Search 1: Exactly 10 boxes (any accuracy)
        search1 = df[df['n_boxes'] == 10].sort_values('accuracy', ascending=False)

        # Search 2: 8-12 boxes with 100% accuracy
        search2 = df[(df['n_boxes'].between(8, 12)) & (df['accuracy'] == 1.0)]

        # Search 3: Highest box count at 100% accuracy
        search3 = df[df['accuracy'] == 1.0].sort_values('n_boxes', ascending=False)

        # Search 4: 9-11 boxes with highest accuracy
        search4 = df[df['n_boxes'].between(9, 11)].sort_values('accuracy', ascending=False)

        if len(search1) > 0:
            print(f"\n  🔍 {model.upper()} - Search 1 (Exactly 10 boxes): {len(search1)} matches")
            for idx, row in search1.head(3).iterrows():
                print(f"     {row['image_name']}: {row['n_boxes']} boxes, {row['n_correct']} correct, {row['accuracy']:.3f} acc")
                all_candidates.append((model, 'search1', row))

        if len(search2) > 0:
            print(f"\n  🔍 {model.upper()} - Search 2 (8-12 boxes, 100% acc): {len(search2)} matches")
            for idx, row in search2.head(3).iterrows():
                print(f"     {row['image_name']}: {row['n_boxes']} boxes, {row['n_correct']} correct, {row['accuracy']:.3f} acc")
                all_candidates.append((model, 'search2', row))

        if len(search3) > 0:
            print(f"\n  🔍 {model.upper()} - Search 3 (Highest box @ 100% acc): {len(search3)} matches")
            for idx, row in search3.head(5).iterrows():
                print(f"     {row['image_name']}: {row['n_boxes']} boxes, {row['n_correct']} correct, {row['accuracy']:.3f} acc")
                if idx < 3:
                    all_candidates.append((model, 'search3', row))

        if len(search4) > 0:
            print(f"\n  🔍 {model.upper()} - Search 4 (9-11 boxes, highest acc): {len(search4)} matches")
            for idx, row in search4.head(3).iterrows():
                print(f"     {row['image_name']}: {row['n_boxes']} boxes, {row['n_correct']} correct, {row['accuracy']:.3f} acc")
                all_candidates.append((model, 'search4', row))

    print("\n" + "-"*100)
    print("  RECOMMENDATION FOR FIGURE 6f")
    print("-"*100)

    if len(all_candidates) > 0:
        print("\n  Best alternatives (prioritized):")
        print("  1. Use highest available box count at 100% accuracy (likely 4-5 boxes)")
        print("  2. Update paper: '5 parasites correctly classified at 100%' instead of '10'")
        print("  3. Alternative: Show 10 boxes with near-perfect accuracy (if available)")
        print("  4. Composite: Show multiple perfect classifications demonstrating reliability")
    else:
        print("\n  ⚠️ No alternatives found.")

    return all_candidates

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*100)
    print("  FOCUSED SEARCH: FIGURE 6c AND 6f ALTERNATIVES")
    print("  Relaxed criteria to find closest matches")
    print("="*100)

    # Search Figure 6c
    candidates_6c = search_figure_6c_alternatives()

    # Search Figure 6f
    candidates_6f = search_figure_6f_alternatives()

    print_section("SEARCH COMPLETE")
    print(f"\n  Figure 6c candidates found: {len(candidates_6c)}")
    print(f"  Figure 6f candidates found: {len(candidates_6f)}")

    if len(candidates_6c) == 0:
        print("\n  ⚠️ WARNING: No suitable alternatives for Figure 6c")
        print("     Recommended action: Manually inspect MP-IDB Stages dataset")

    if len(candidates_6f) == 0:
        print("\n  ⚠️ WARNING: No suitable alternatives for Figure 6f")
        print("     Recommended action: Use highest available box count at 100% accuracy")

    print("\n" + "="*100)
