#!/usr/bin/env python3
"""
Copy Selected Paper Images to Publication Folder

Based on IMAGE_SELECTION_REPORT.md analysis, this script copies
the carefully selected images for Figures 3 and 4 to the paper folder.

Selected images tell a compelling story:
- (a) Baseline success
- (b) High density success
- (c) Ultra-rare class success (HERO)
- (d) Realistic challenge (HONEST)
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

# Selected images configuration
SELECTED_IMAGES = [
    {
        "name": "Figure 3a: IML Lifecycle Detection",
        "source": RESULTS_DIR / "experiment_iml_lifecycle" / "visualizations" / "pred_detection_yolo11" / "PA171772.png",
        "dest": OUTPUT_DIR / "iml_lifecycle_detection.png",
        "metrics": "4/4 boxes, conf=0.859, Perfect"
    },
    {
        "name": "Figure 4a: IML Lifecycle Classification",
        "source": RESULTS_DIR / "experiment_iml_lifecycle" / "visualizations" / "pred_classification_efficientnet_b1_focal" / "PA171772.png",
        "dest": OUTPUT_DIR / "iml_lifecycle_classification.png",
        "metrics": "4/4 correct, conf=0.881, All correct"
    },
    {
        "name": "Figure 3b: MP-IDB Species Detection",
        "source": RESULTS_DIR / "experiment_mp_idb_species" / "visualizations" / "pred_detection_yolo11" / "1409171742-0010-R.png",
        "dest": OUTPUT_DIR / "mp_idb_species_detection.png",
        "metrics": "5/5 boxes, conf=0.727, Perfect (High Density)"
    },
    {
        "name": "Figure 4b: MP-IDB Species Classification",
        "source": RESULTS_DIR / "experiment_mp_idb_species" / "visualizations" / "pred_classification_efficientnet_b1_focal" / "1409171742-0010-R.png",
        "dest": OUTPUT_DIR / "mp_idb_species_classification.png",
        "metrics": "5/5 correct, conf=0.925, All correct"
    },
    {
        "name": "Figure 3c: MP-IDB Stages Detection (HERO - Ultra-Rare Gametocyte)",
        "source": RESULTS_DIR / "experiment_mp_idb_stages" / "visualizations" / "pred_detection_yolo11" / "1703121298-0010-G.png",
        "dest": OUTPUT_DIR / "mp_idb_stages_detection.png",
        "metrics": "2/2 gametocytes, conf=0.852, Perfect (1.7% of data, 54:1 imbalance)"
    },
    {
        "name": "Figure 4c: MP-IDB Stages Classification (HERO)",
        "source": RESULTS_DIR / "experiment_mp_idb_stages" / "visualizations" / "pred_classification_resnet50_focal" / "1703121298-0010-G.png",
        "dest": OUTPUT_DIR / "mp_idb_stages_classification.png",
        "metrics": "2/2 gametocytes correct, conf=0.986, All correct"
    },
    {
        "name": "Figure 3d: MD_2019 Stages Detection (HONEST - Realistic Challenge)",
        "source": RESULTS_DIR / "experiment_md_2019_stages" / "visualizations" / "pred_detection_yolo11" / "Trip 064 Day 2 25-11-05 Image 5_11.png",
        "dest": OUTPUT_DIR / "md_2019_stages_detection.png",
        "metrics": "7/8 boxes (1 FN), conf=0.828, Missing detections"
    },
    {
        "name": "Figure 4d: MD_2019 Stages Classification (HONEST)",
        "source": RESULTS_DIR / "experiment_md_2019_stages" / "visualizations" / "pred_classification_efficientnet_b0_focal" / "Trip 064 Day 2 25-11-05 Image 5_11.png",
        "dest": OUTPUT_DIR / "md_2019_stages_classification.png",
        "metrics": "7/8 correct (1 wrong), conf=0.744, Mixed"
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
    print("[IMAGE COPY] COPYING SELECTED IMAGES TO PAPER FOLDER")
    print("="*80)
    print(f"Destination: {OUTPUT_DIR}")
    print(f"Total images to copy: {len(SELECTED_IMAGES)}")

    # Backup existing files first
    backup_existing_files()

    print("\n[COPY] Starting image copy process...\n")

    success_count = 0
    failed_count = 0

    for img_config in SELECTED_IMAGES:
        print(f"[{img_config['name']}]")
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
        print("\nNext steps:")
        print("   1. Verify images visually in the output folder")
        print("   2. Review IMAGE_SELECTION_REPORT.md for detailed metrics")
        print("   3. Update paper draft with exact figure captions")
        return True
    else:
        print("\n⚠️ SOME IMAGES FAILED TO COPY!")
        print("   Check error messages above and verify source paths")
        return False


def verify_paper_references():
    """Verify that paper draft references match our image filenames"""

    paper_draft = BASE_DIR / "luaran" / "templates" / "KINETIK_PAPER_DRAFT_UPDATED_2025.md"

    if not paper_draft.exists():
        print("\n[WARNING] Paper draft not found for verification")
        return

    print("\n" + "="*80)
    print("[VERIFY] CHECKING PAPER REFERENCES")
    print("="*80)

    expected_references = [
        "iml_lifecycle_detection.png",
        "iml_lifecycle_classification.png",
        "mp_idb_species_detection.png",
        "mp_idb_species_classification.png",
        "mp_idb_stages_detection.png",
        "mp_idb_stages_classification.png",
        "md_2019_stages_detection.png",
        "md_2019_stages_classification.png",
    ]

    with open(paper_draft, 'r', encoding='utf-8') as f:
        paper_content = f.read()

    all_found = True
    for ref in expected_references:
        if ref in paper_content:
            print(f"   ✅ Found reference: {ref}")
        else:
            print(f"   ⚠️ Missing reference: {ref}")
            all_found = False

    if all_found:
        print("\n✅ All image references found in paper draft!")
    else:
        print("\n⚠️ Some image references not found in paper draft")
        print("   → Paper draft may need updating")


def main():
    """Main execution"""

    print("\n" + "="*80)
    print("SELECTED PAPER IMAGES COPY SCRIPT")
    print("="*80)
    print("Based on: IMAGE_SELECTION_REPORT.md")
    print("Strategy: 2×2×2 layout with compelling narrative")
    print("Story Arc: Perfect → High Density → Ultra-Rare Hero → Realistic Challenge")

    # Copy images
    success = copy_selected_images()

    # Verify paper references
    verify_paper_references()

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
