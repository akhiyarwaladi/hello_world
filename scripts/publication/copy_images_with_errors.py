#!/usr/bin/env python3
"""
Copy Selected Images with ERRORS for 3×2 Grid Layout

ALL images must have errors (no perfect green boxes).
- 3 Columns: SMALL ERROR → MEDIUM ERROR → LARGE ERROR
- 2 Rows: Detection → Classification
- Total: 6 images (all showing model limitations)

Strategy: Show honest scientific assessment with error progression.
"""

import shutil
from pathlib import Path
from datetime import datetime


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "optA_20251016_200330" / "experiments"
OUTPUT_DIR = BASE_DIR / "luaran" / "templates" / "figures" / "qualitative_analysis"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Selected images configuration (3×2 grid - ALL WITH ERRORS)
SELECTED_IMAGES = [
    # COLUMN 1: SMALL ERROR (MP-IDB Species)
    {
        "name": "Column 1 Row 1: MP-IDB Species Detection (SMALL ERROR)",
        "source": RESULTS_DIR / "experiment_mp_idb_species" / "visualizations" / "pred_detection_yolo11" / "1704282807-0010-R.png",
        "dest": OUTPUT_DIR / "c1_r1_species_detection_error.png",
        "metrics": "GT:14, Pred:15 (2 FP + 1 FN), conf=0.746",
        "caption": "Small Error - Mixed FP and FN"
    },
    {
        "name": "Column 1 Row 2: MP-IDB Species Classification (SMALL ERROR)",
        "source": RESULTS_DIR / "experiment_mp_idb_species" / "visualizations" / "pred_classification_efficientnet_b1_focal" / "1703121298-0008-G.png",
        "dest": OUTPUT_DIR / "c1_r2_species_classification_error.png",
        "metrics": "1 box: 0 correct, 1 wrong (0% accuracy), conf=0.782",
        "caption": "Small Error - 1 wrong classification"
    },

    # COLUMN 2: MEDIUM ERROR (MP-IDB Stages)
    {
        "name": "Column 2 Row 1: MP-IDB Stages Detection (MEDIUM ERROR)",
        "source": RESULTS_DIR / "experiment_mp_idb_stages" / "visualizations" / "pred_detection_yolo11" / "1701151546-0012-R.png",
        "dest": OUTPUT_DIR / "c2_r1_stages_detection_error.png",
        "metrics": "GT:22, Pred:21 (0 FP + 1 FN), conf=0.770",
        "caption": "Medium Error - 1 missed detection on high-density field"
    },
    {
        "name": "Column 2 Row 2: MP-IDB Stages Classification (MEDIUM ERROR)",
        "source": RESULTS_DIR / "experiment_mp_idb_stages" / "visualizations" / "pred_classification_resnet50_focal" / "1701151546-0006-R_T.png",
        "dest": OUTPUT_DIR / "c2_r2_stages_classification_error.png",
        "metrics": "19 boxes: 17 correct, 2 wrong (89.5% accuracy), conf=0.819",
        "caption": "Medium Error - 2 misclassifications"
    },

    # COLUMN 3: LARGE ERROR (MD_2019)
    {
        "name": "Column 3 Row 1: MD_2019 Detection (LARGE ERROR)",
        "source": RESULTS_DIR / "experiment_md_2019_stages" / "visualizations" / "pred_detection_yolo11" / "Trip 064 Day 2 25-11-05 Image 7_8.png",
        "dest": OUTPUT_DIR / "c3_r1_md2019_detection_error.png",
        "metrics": "GT:3, Pred:1 (0 FP + 2 FN), conf=0.904",
        "caption": "Large Error - 2 missed detections (66% miss rate)"
    },
    {
        "name": "Column 3 Row 2: MD_2019 Classification (LARGE ERROR)",
        "source": RESULTS_DIR / "experiment_md_2019_stages" / "visualizations" / "pred_classification_efficientnet_b0_focal" / "Trip 029 Day 1 22-11-05 Image 11 add_1.png",
        "dest": OUTPUT_DIR / "c3_r2_md2019_classification_error.png",
        "metrics": "5 boxes: 2 correct, 3 wrong (40% accuracy), conf=0.802",
        "caption": "Large Error - 3 misclassifications (60% error rate)"
    },
]


def copy_selected_images():
    """Copy all selected images to paper folder"""

    print("\n" + "="*80)
    print("[IMAGE COPY] COPYING IMAGES WITH ERRORS FOR 3×2 LAYOUT")
    print("="*80)
    print(f"Destination: {OUTPUT_DIR}")
    print(f"Layout: 3 columns × 2 rows = 6 images")
    print("Columns: SMALL ERROR → MEDIUM ERROR → LARGE ERROR")
    print("Strategy: Show honest scientific assessment\n")

    success_count = 0
    failed_count = 0

    for img_config in SELECTED_IMAGES:
        print(f"[{img_config['name']}]")
        print(f"   Caption: {img_config['caption']}")
        print(f"   Metrics: {img_config['metrics']}")

        if not img_config['source'].exists():
            print(f"   ❌ ERROR: Source file not found!")
            print(f"      Expected: {img_config['source']}")
            failed_count += 1
            continue

        try:
            # Copy file
            shutil.copy2(img_config['source'], img_config['dest'])

            # Verify copy
            if img_config['dest'].exists():
                source_size = img_config['source'].stat().st_size
                dest_size = img_config['dest'].stat().st_size

                if source_size == dest_size:
                    print(f"   ✅ SUCCESS: Copied ({dest_size:,} bytes)")
                    print(f"      → {img_config['dest'].name}")
                    success_count += 1
                else:
                    print(f"   ⚠️ WARNING: File sizes don't match!")
                    failed_count += 1
            else:
                print(f"   ❌ ERROR: Copy failed (destination not found)")
                failed_count += 1

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            failed_count += 1

        print()

    # Summary
    print("="*80)
    print("[SUMMARY] IMAGE COPY COMPLETE")
    print("="*80)
    print(f"   ✅ Successful: {success_count}/{len(SELECTED_IMAGES)}")
    print(f"   ❌ Failed: {failed_count}/{len(SELECTED_IMAGES)}")
    print(f"   📁 Output folder: {OUTPUT_DIR}")

    if success_count == len(SELECTED_IMAGES):
        print("\n✨ ALL IMAGES COPIED SUCCESSFULLY! ✨")
        print("\n3×2 Layout Structure (ALL WITH ERRORS):")
        print("   Column 1: SMALL ERROR (MP-IDB Species)")
        print("   Column 2: MEDIUM ERROR (MP-IDB Stages)")
        print("   Column 3: LARGE ERROR (MD_2019)")
        print("\nNext steps:")
        print("   1. Verify images visually - should see red/yellow boxes")
        print("   2. Update paper draft with error-focused narrative")
        print("   3. Write captions emphasizing limitations and honest assessment")
        return True
    else:
        print("\n⚠️ SOME IMAGES FAILED TO COPY!")
        return False


def print_layout_summary():
    """Print visual summary of 3×2 layout with errors"""

    print("\n" + "="*80)
    print("[LAYOUT] 3×2 GRID STRUCTURE (ALL WITH ERRORS)")
    print("="*80)

    print("\n┌─────────────────────┬─────────────────────┬─────────────────────┐")
    print("│ Column 1: SMALL ERR │ Column 2: MEDIUM ERR│ Column 3: LARGE ERR │")
    print("│ MP-IDB Species      │ MP-IDB Stages       │ MD_2019 Stages      │")
    print("├─────────────────────┼─────────────────────┼─────────────────────┤")
    print("│ Row 1: Detection    │ Row 1: Detection    │ Row 1: Detection    │")
    print("│ GT:14 → Pred:15     │ GT:22 → Pred:21     │ GT:3 → Pred:1       │")
    print("│ 2 FP + 1 FN         │ 1 FN                │ 2 FN (66% miss)     │")
    print("│ conf=0.746          │ conf=0.770          │ conf=0.904          │")
    print("├─────────────────────┼─────────────────────┼─────────────────────┤")
    print("│ Row 2: Classification│ Row 2: Classification│ Row 2: Classification│")
    print("│ 1 box: 0/1          │ 19 boxes: 17/19     │ 5 boxes: 2/5        │")
    print("│ 100% wrong          │ 89.5% accuracy      │ 40% accuracy        │")
    print("│ conf=0.782          │ conf=0.819          │ conf=0.802          │")
    print("└─────────────────────┴─────────────────────┴─────────────────────┘")

    print("\nNarrative Flow:")
    print("   → Column 1: Minor errors (FP/FN on high-density, 1 misclassification)")
    print("   → Column 2: Moderate errors (1 FN on 22 parasites, 2 misclassifications)")
    print("   → Column 3: Major errors (66% FN rate, 60% misclassification rate)")
    print("\nKey Message:")
    print("   Demonstrates honest scientific assessment of system limitations")
    print("   Shows error progression from minor to major challenges")
    print("   Provides basis for Limitations and Future Work discussion")


def main():
    """Main execution"""

    print("\n" + "="*80)
    print("PAPER IMAGES WITH ERRORS - 3×2 LAYOUT")
    print("="*80)
    print("Layout: 3 columns × 2 rows = 6 images")
    print("Strategy: Show honest error progression")
    print("ALL IMAGES HAVE ERRORS - No perfect cases")

    # Print layout structure
    print_layout_summary()

    # Copy images
    success = copy_selected_images()

    if success:
        print("\n" + "="*80)
        print("✨ PROCESS COMPLETE - READY FOR HONEST PAPER! ✨")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️ PROCESS COMPLETED WITH ERRORS")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit(main())
