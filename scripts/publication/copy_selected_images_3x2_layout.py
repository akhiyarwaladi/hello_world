#!/usr/bin/env python3
"""
Copy Selected Images for 3×2 Grid Layout

Based on Option A: 3×2 Grid (6 images, compact)
- 3 Columns: EASY → ULTRA-RARE HERO → REALISTIC CHALLENGE
- 2 Rows: Detection → Classification
- Total: 6 images (more compact than 2×2×2 layout)

Selected images tell a compelling story:
- Column 1: Baseline success (IML 4/4)
- Column 2: Ultra-rare gametocyte success (MP-IDB 2/2, conf=0.986) 🌟
- Column 3: Realistic challenge (MD_2019 7/8)
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

# Selected images configuration (3×2 grid)
SELECTED_IMAGES = [
    # COLUMN 1: EASY (IML Lifecycle)
    {
        "name": "Column 1 Row 1: IML Lifecycle Detection (EASY)",
        "source": RESULTS_DIR / "experiment_iml_lifecycle" / "visualizations" / "pred_detection_yolo11" / "PA171772.png",
        "dest": OUTPUT_DIR / "c1_r1_iml_lifecycle_detection.png",
        "metrics": "4/4 boxes, conf=0.859, Perfect",
        "caption": "Easy - Perfect 4/4 detection"
    },
    {
        "name": "Column 1 Row 2: IML Lifecycle Classification (EASY)",
        "source": RESULTS_DIR / "experiment_iml_lifecycle" / "visualizations" / "pred_classification_efficientnet_b1_focal" / "PA171772.png",
        "dest": OUTPUT_DIR / "c1_r2_iml_lifecycle_classification.png",
        "metrics": "4/4 correct, conf=0.881, All correct",
        "caption": "Easy - Perfect 4/4 classification"
    },

    # COLUMN 2: ULTRA-RARE HERO (MP-IDB Stages - Gametocyte)
    {
        "name": "Column 2 Row 1: MP-IDB Stages Detection (ULTRA-RARE HERO)",
        "source": RESULTS_DIR / "experiment_mp_idb_stages" / "visualizations" / "pred_detection_yolo11" / "1703121298-0010-G.png",
        "dest": OUTPUT_DIR / "c2_r1_mp_idb_stages_detection.png",
        "metrics": "2/2 gametocytes, conf=0.852, Perfect (1.7% of data, 54:1 imbalance)",
        "caption": "Ultra-Rare Hero - Perfect 2/2 gametocyte detection"
    },
    {
        "name": "Column 2 Row 2: MP-IDB Stages Classification (ULTRA-RARE HERO)",
        "source": RESULTS_DIR / "experiment_mp_idb_stages" / "visualizations" / "pred_classification_resnet50_focal" / "1703121298-0010-G.png",
        "dest": OUTPUT_DIR / "c2_r2_mp_idb_stages_classification.png",
        "metrics": "2/2 gametocytes correct, conf=0.986, All correct 🌟",
        "caption": "Ultra-Rare Hero - Perfect 2/2 with 0.986 confidence"
    },

    # COLUMN 3: REALISTIC CHALLENGE (MD_2019)
    {
        "name": "Column 3 Row 1: MD_2019 Stages Detection (REALISTIC CHALLENGE)",
        "source": RESULTS_DIR / "experiment_md_2019_stages" / "visualizations" / "pred_detection_yolo11" / "Trip 064 Day 2 25-11-05 Image 5_11.png",
        "dest": OUTPUT_DIR / "c3_r1_md_2019_stages_detection.png",
        "metrics": "7/8 boxes (1 FN), conf=0.828, Missing detections",
        "caption": "Realistic Challenge - 7/8 detection (1 false negative)"
    },
    {
        "name": "Column 3 Row 2: MD_2019 Stages Classification (REALISTIC CHALLENGE)",
        "source": RESULTS_DIR / "experiment_md_2019_stages" / "visualizations" / "pred_classification_efficientnet_b0_focal" / "Trip 064 Day 2 25-11-05 Image 5_11.png",
        "dest": OUTPUT_DIR / "c3_r2_md_2019_stages_classification.png",
        "metrics": "7/8 correct (1 wrong), conf=0.744, Mixed",
        "caption": "Realistic Challenge - 7/8 classification with lower confidence"
    },
]


def backup_existing_files():
    """Backup existing files if they exist"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = OUTPUT_DIR.parent / f"qualitative_analysis_backup_{timestamp}"

    existing_files = list(OUTPUT_DIR.glob("*.png"))
    if existing_files:
        print(f"\n[BACKUP] Found {len(existing_files)} existing files")
        backup_dir.mkdir(parents=True, exist_ok=True)

        for file in existing_files:
            backup_path = backup_dir / file.name
            shutil.copy2(file, backup_path)
            print(f"   ✓ Backed up: {file.name}")

        print(f"[BACKUP] All existing files saved to: {backup_dir.name}")
        return True

    return False


def copy_selected_images():
    """Copy all selected images to paper folder"""

    print("\n" + "="*80)
    print("[IMAGE COPY] COPYING SELECTED IMAGES FOR 3×2 LAYOUT")
    print("="*80)
    print(f"Destination: {OUTPUT_DIR}")
    print(f"Layout: 3 columns × 2 rows = 6 images")
    print("Columns: EASY → ULTRA-RARE HERO → REALISTIC CHALLENGE")
    print("Rows: Detection → Classification")

    # Backup existing files first
    backup_existing_files()

    print("\n[COPY] Starting image copy process...\n")

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
                    print(f"      Source: {source_size:,} bytes")
                    print(f"      Dest: {dest_size:,} bytes")
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
        print("\n3×2 Layout Structure:")
        print("   Column 1: EASY (IML Lifecycle)")
        print("   Column 2: ULTRA-RARE HERO (MP-IDB Stages - Gametocyte 0.986 conf)")
        print("   Column 3: REALISTIC CHALLENGE (MD_2019 7/8)")
        print("\nNext steps:")
        print("   1. Verify images visually in the output folder")
        print("   2. Update paper draft with 3×2 figure layout")
        print("   3. Update figure captions with exact metrics")
        return True
    else:
        print("\n⚠️ SOME IMAGES FAILED TO COPY!")
        print("   Check error messages above and verify source paths")
        return False


def print_layout_summary():
    """Print visual summary of 3×2 layout"""

    print("\n" + "="*80)
    print("[LAYOUT] 3×2 GRID STRUCTURE")
    print("="*80)

    print("\n┌─────────────────────┬─────────────────────┬─────────────────────┐")
    print("│  Column 1: EASY     │ Column 2: ULTRA-RARE│ Column 3: REALISTIC │")
    print("│  IML Lifecycle      │ MP-IDB Stages       │ MD_2019 Stages      │")
    print("├─────────────────────┼─────────────────────┼─────────────────────┤")
    print("│ Row 1: Detection    │ Row 1: Detection    │ Row 1: Detection    │")
    print("│ 4/4 Perfect         │ 2/2 Gametocytes     │ 7/8 (1 FN)          │")
    print("│ conf=0.859          │ conf=0.852          │ conf=0.828          │")
    print("├─────────────────────┼─────────────────────┼─────────────────────┤")
    print("│ Row 2: Classification│ Row 2: Classification│ Row 2: Classification│")
    print("│ 4/4 Correct         │ 2/2 Correct 🌟      │ 7/8 (1 wrong)       │")
    print("│ conf=0.881          │ conf=0.986          │ conf=0.744          │")
    print("└─────────────────────┴─────────────────────┴─────────────────────┘")

    print("\nNarrative Flow:")
    print("   → Column 1: Builds confidence (perfect baseline)")
    print("   → Column 2: Highlights key contribution (ultra-rare success with 0.986 conf)")
    print("   → Column 3: Shows honest limitations (realistic 7/8 performance)")


def main():
    """Main execution"""

    print("\n" + "="*80)
    print("SELECTED PAPER IMAGES COPY SCRIPT - 3×2 LAYOUT")
    print("="*80)
    print("Layout: 3 columns × 2 rows = 6 images (more compact)")
    print("Strategy: Difficulty Progression")
    print("Story Arc: Easy → Ultra-Rare Hero → Realistic Challenge")

    # Print layout structure
    print_layout_summary()

    # Copy images
    success = copy_selected_images()

    if success:
        print("\n" + "="*80)
        print("✨ PROCESS COMPLETE - READY FOR PAPER! ✨")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️ PROCESS COMPLETED WITH ERRORS")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit(main())
