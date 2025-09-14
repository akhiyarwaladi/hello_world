#!/usr/bin/env python3
"""
Debug Species Mapping in Crop Detection Script
Investigate why only 'parasite' class is generated instead of species-specific classes
"""

import csv
from pathlib import Path
from typing import Dict

MPIDB_ROOT = Path("data/raw/mp_idb")
SPECIES_DIRS = ["Falciparum", "Vivax", "Malariae", "Ovale"]
SPECIES_CANON = {
    "Falciparum": "falciparum",
    "Vivax": "vivax",
    "Malariae": "malariae",
    "Ovale": "ovale",
}

def debug_species_mapping():
    """Debug species mapping process step by step"""

    print("🔍 DEBUGGING SPECIES MAPPING")
    print("=" * 50)

    # Check if MP-IDB root exists
    if not MPIDB_ROOT.exists():
        print(f"❌ MP-IDB root not found: {MPIDB_ROOT}")
        return

    print(f"✅ MP-IDB root found: {MPIDB_ROOT}")

    # Check each species directory
    mapping = {}
    for sp in SPECIES_DIRS:
        print(f"\n📁 Processing species: {sp}")

        csv_path = MPIDB_ROOT / sp / f"mp-idb-{sp.lower()}.csv"
        alt_path = MPIDB_ROOT / sp / f"mp-idb-{sp.lower()}-abspath.csv"

        print(f"   Looking for: {csv_path}")
        print(f"   Alternative: {alt_path}")

        if csv_path.exists():
            print(f"   ✅ Found: {csv_path}")
            actual_path = csv_path
        elif alt_path.exists():
            print(f"   ✅ Found alternative: {alt_path}")
            actual_path = alt_path
        else:
            print(f"   ❌ Neither CSV found for {sp}")
            continue

        # Read CSV and build mapping
        try:
            with open(actual_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                species_count = 0
                for row in reader:
                    fn = Path(row.get("filename", "")).name
                    if fn:
                        stem = Path(fn).stem
                        mapping[stem] = SPECIES_CANON[sp]
                        species_count += 1

                        # Show first few mappings
                        if species_count <= 3:
                            print(f"      {stem} -> {SPECIES_CANON[sp]}")

                print(f"   📊 Total mappings for {sp}: {species_count}")

        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")

    print(f"\n📈 SUMMARY:")
    print(f"   Total filename mappings: {len(mapping)}")

    # Show species distribution
    species_counts = {}
    for species in mapping.values():
        species_counts[species] = species_counts.get(species, 0) + 1

    print(f"   Species distribution:")
    for species, count in species_counts.items():
        print(f"     - {species}: {count} images")

    # Check current detection dataset
    print(f"\n🔍 CHECKING DETECTION DATASET:")
    det_images_dir = Path("data/detection/images")
    if det_images_dir.exists():
        det_images = list(det_images_dir.glob("*.jpg"))
        print(f"   Detection images found: {len(det_images)}")

        # Check how many have species mapping
        mapped_count = 0
        unmapped_examples = []
        for img_path in det_images[:10]:  # Check first 10
            stem = img_path.stem
            if stem in mapping:
                mapped_count += 1
                species = mapping[stem]
                print(f"     ✅ {stem} -> {species}")
            else:
                unmapped_examples.append(stem)

        if unmapped_examples:
            print(f"   ❌ Unmapped examples:")
            for example in unmapped_examples[:3]:
                print(f"     - {example}")

        print(f"   Mapping coverage: {mapped_count}/{min(10, len(det_images))}")
    else:
        print(f"   ❌ Detection images directory not found: {det_images_dir}")

    return mapping

def test_crop_process():
    """Test actual cropping process to see what goes wrong"""

    print(f"\n🧪 TESTING CROP PROCESS:")

    # Check if classification_crops already exists
    crops_dir = Path("data/classification_crops")
    if crops_dir.exists():
        print(f"   Current classification dataset structure:")
        for split in ['train', 'val', 'test']:
            split_dir = crops_dir / split
            if split_dir.exists():
                classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
                print(f"     {split}: {classes}")

    # Check crops_index.json if it exists
    index_file = crops_dir / "crops_index.json"
    if index_file.exists():
        print(f"\n📋 CHECKING CROPS INDEX:")
        try:
            import json
            with open(index_file, 'r') as f:
                index = json.load(f)

            for split, items in index.items():
                if items:
                    labels = [item.get('label', 'unknown') for item in items]
                    label_counts = {}
                    for label in labels:
                        label_counts[label] = label_counts.get(label, 0) + 1

                    print(f"     {split} split labels: {label_counts}")

        except Exception as e:
            print(f"   ❌ Error reading index: {e}")

if __name__ == "__main__":
    print("🚀 SPECIES MAPPING DEBUG ANALYSIS")
    print("=" * 60)

    # Debug species mapping
    mapping = debug_species_mapping()

    # Test crop process
    test_crop_process()

    print(f"\n💡 ANALYSIS COMPLETE")
    print("=" * 60)