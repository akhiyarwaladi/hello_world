#!/usr/bin/env python3
"""
Fix classification structure - reorganize crops by ground truth class
"""

import pandas as pd
import shutil
import argparse
from pathlib import Path
import yaml

def detect_dataset_type(input_dir):
    """Detect dataset type from input directory path"""
    input_path = str(input_dir).lower()

    if "processed/species" in input_path:
        return "mp_idb_species"
    elif "processed/stages" in input_path:
        return "mp_idb_stages"
    elif "processed/lifecycle" in input_path:
        return "iml_lifecycle"
    else:
        # Try to detect from data.yaml if exists
        potential_yaml = Path(input_dir) / "data.yaml"
        if potential_yaml.exists():
            try:
                with open(potential_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                names = data.get('names', [])
                if len(names) == 1 and names[0] in ['parasite', 'parasit']:
                    return "mp_idb_species"
                elif len(names) == 4 and 'ring' in names:
                    if 'schizont' in names:
                        return "mp_idb_stages" if 'trophozoite' in names else "iml_lifecycle"
                    return "iml_lifecycle"
                elif len(names) == 5 and 'red_blood_cell' in names:
                    return "iml_lifecycle"
            except:
                pass
        return "mp_idb_species"  # Default fallback

def load_class_names_by_dataset(input_dir):
    """Load class names based on detected dataset type"""
    dataset_type = detect_dataset_type(input_dir)

    if dataset_type == "mp_idb_species":
        return {0: "P_falciparum", 1: "P_vivax", 2: "P_malariae", 3: "P_ovale"}
    elif dataset_type == "mp_idb_stages":
        return {0: "ring", 1: "schizont", 2: "trophozoite", 3: "gametocyte"}
    elif dataset_type == "iml_lifecycle":
        return {0: "red_blood_cell", 1: "ring", 2: "gametocyte", 3: "trophozoite", 4: "schizont"}
    else:
        # Fallback to species
        return {0: "P_falciparum", 1: "P_vivax", 2: "P_malariae", 3: "P_ovale"}

def get_ground_truth_class(image_path, input_path):
    """Get ground truth class from label file"""
    # Convert image path to label path
    label_path = image_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
    label_file = Path(input_path) / label_path

    if label_file.exists():
        with open(label_file, 'r') as f:
            line = f.readline().strip()
            if line:
                class_id = int(line.split()[0])
                return class_id
    return None

def main():
    parser = argparse.ArgumentParser(description="Fix classification structure for dataset-specific classes")
    parser.add_argument("--crop_data_path", required=True,
                       help="Path to crop data directory")
    parser.add_argument("--input_path", required=True,
                       help="Path to input YOLO dataset directory")

    args = parser.parse_args()

    # Detect dataset type and load appropriate class mapping
    dataset_type = detect_dataset_type(args.input_path)
    class_names = load_class_names_by_dataset(args.input_path)

    print(f"[INFO] Detected dataset type: {dataset_type}")
    print(f"[INFO] Using class mapping: {class_names}")

    # Paths
    crops_base = Path(args.crop_data_path)
    metadata_file = crops_base / "crop_metadata.csv"
    yolo_class_dir = crops_base / "yolo_classification"
    input_path = Path(args.input_path)

    # Read metadata
    df = pd.read_csv(metadata_file)

    print(f"Processing {len(df)} crops...")

    # Add ground truth class column
    df['ground_truth_class'] = df['original_image'].apply(
        lambda x: get_ground_truth_class(x, input_path)
    )

    # Show class distribution
    class_dist = df['ground_truth_class'].value_counts()
    print(f"\nClass distribution:")
    for class_id, count in class_dist.items():
        if class_id is not None:
            print(f"   {class_names.get(class_id, f'Unknown_{class_id}')}: {count} crops")

    # Remove old single-class structure
    if yolo_class_dir.exists():
        shutil.rmtree(yolo_class_dir)

    # Create new multi-class structure
    for split in ['train', 'val', 'test']:
        split_data = df[df['split'] == split]

        for class_id in split_data['ground_truth_class'].unique():
            if class_id is not None:
                class_name = class_names.get(class_id, f"unknown_{class_id}")
                class_dir = yolo_class_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

        # Copy files to appropriate class folders
        for _, row in split_data.iterrows():
            if row['ground_truth_class'] is not None:
                class_name = class_names.get(row['ground_truth_class'], f"unknown_{row['ground_truth_class']}")

                src_file = crops_base / "crops" / split / row['crop_filename']
                dst_file = yolo_class_dir / split / class_name / row['crop_filename']

                if src_file.exists():
                    shutil.copy2(src_file, dst_file)

    # Save updated metadata
    df.to_csv(metadata_file, index=False)

    print(f"\n[SUCCESS] Classification structure fixed!")
    print(f"[STRUCTURE] New structure: {yolo_class_dir}")

    # Show final structure
    for split in ['train', 'val', 'test']:
        split_dir = yolo_class_dir / split
        if split_dir.exists():
            print(f"\n{split.upper()}:")
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.jpg")))
                    print(f"   {class_dir.name}: {count} images")

if __name__ == "__main__":
    main()