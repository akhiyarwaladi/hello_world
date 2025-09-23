#!/usr/bin/env python3
"""
Fix classification structure - reorganize crops by ground truth class
"""

import pandas as pd
import shutil
import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Fix classification structure for 4 malaria species")
    parser.add_argument("--crop_data_path", required=True,
                       help="Path to crop data directory")
    parser.add_argument("--input_path", required=True,
                       help="Path to input YOLO dataset directory")

    args = parser.parse_args()

    # Class mapping
    class_names = {
        0: "P_falciparum",
        1: "P_malariae",
        2: "P_ovale",
        3: "P_vivax"
    }

    # Paths
    crops_base = Path(args.crop_data_path)
    metadata_file = crops_base / "crop_metadata.csv"
    yolo_class_dir = crops_base / "yolo_classification"
    input_path = Path(args.input_path)

    # Read metadata
    df = pd.read_csv(metadata_file)

    print(f"📊 Processing {len(df)} crops...")

    # Add ground truth class column
    df['ground_truth_class'] = df['original_image'].apply(
        lambda x: get_ground_truth_class(x, input_path)
    )

    # Show class distribution
    class_dist = df['ground_truth_class'].value_counts()
    print(f"\n📈 Class distribution:")
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

    print(f"\n✅ Classification structure fixed!")
    print(f"📂 New structure: {yolo_class_dir}")

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