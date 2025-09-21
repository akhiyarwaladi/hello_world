#!/usr/bin/env python3
"""
Generate Classification Crops from Detection Model Results
This script uses a trained detection model to detect parasites and crop them for classification training
"""

import os
import sys
import argparse
import cv2
import numpy as np
import subprocess
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

def load_class_names(data_yaml_path="data/integrated/yolo/data.yaml"):
    """Load class names from YOLO data.yaml file"""
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('names', [])
    except:
        # Default class names if file not found
        return ["P_falciparum", "P_vivax", "P_malariae", "P_ovale"]

def get_ground_truth_class(image_path, input_dir):
    """Get ground truth class for an image based on YOLO label file"""
    try:
        # Convert image path to corresponding label path
        input_path = Path(input_dir)
        image_rel_path = Path(image_path).relative_to(input_path)

        # Handle different split structures
        if 'images' in image_rel_path.parts:
            # Replace 'images' with 'labels' and change extension
            label_parts = list(image_rel_path.parts)
            for i, part in enumerate(label_parts):
                if part == 'images':
                    label_parts[i] = 'labels'
                    break
            label_rel_path = Path(*label_parts).with_suffix('.txt')
        else:
            # Direct path without images folder
            label_rel_path = image_rel_path.with_suffix('.txt')

        label_path = input_path / label_rel_path

        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            if lines:
                # Get first class (assuming single class per image for most cases)
                first_line = lines[0].strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    return class_id

        # Default class if no label found
        return 0
    except Exception as e:
        print(f"Warning: Could not get ground truth for {image_path}: {e}")
        return 0

def load_detection_model(model_path):
    """Load trained detection model"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"🔄 Loading detection model: {model_path}")
    model = YOLO(model_path)
    return model

def detect_and_crop(model, image_path, confidence=0.25, crop_size=128):
    """Detect parasites in image and return crops"""
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    # Run detection
    results = model(image, conf=confidence, verbose=False)

    crops = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Calculate center and expand to square crop
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Create square crop around center
                half_size = crop_size // 2
                crop_x1 = max(0, center_x - half_size)
                crop_y1 = max(0, center_y - half_size)
                crop_x2 = min(image.shape[1], center_x + half_size)
                crop_y2 = min(image.shape[0], center_y + half_size)

                # Extract crop
                crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

                # Resize to exact crop size if needed
                if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                    crop = cv2.resize(crop, (crop_size, crop_size))

                crops.append({
                    'crop': crop,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'crop_coords': [crop_x1, crop_y1, crop_x2, crop_y2]
                })

    return crops

def process_dataset(model, input_dir, output_dir, dataset_name, confidence=0.25, crop_size=128):
    """Process entire dataset and generate crops"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Load class names for PyTorch structure
    class_names = load_class_names()
    print(f"📋 Using class names: {class_names}")

    # Create output directories
    crops_dir = output_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Create PyTorch ImageFolder structure: train/val/test -> class_name subfolders
    for split in ['train', 'val', 'test']:
        split_path = input_path / split
        if split_path.exists():
            split_dir = crops_dir / split
            split_dir.mkdir(exist_ok=True)
            # Create class subdirectories for PyTorch ImageFolder format
            for class_name in class_names:
                (split_dir / class_name).mkdir(exist_ok=True)

    # Process images and collect metadata
    metadata = []
    processed_count = 0
    crop_count = 0

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    all_images = []

    for split in ['train', 'val', 'test']:
        split_path = input_path / split / "images"
        if split_path.exists():
            for ext in image_extensions:
                all_images.extend([(img, split) for img in split_path.glob(ext)])

    # If no split structure, process all images in input_dir
    if not all_images:
        for ext in image_extensions:
            all_images.extend([(img, 'all') for img in input_path.glob(f"**/{ext}")])

    print(f"📊 Found {len(all_images)} images to process")

    # Process each image
    for image_path, split in tqdm(all_images, desc="Generating crops"):
        try:
            crops = detect_and_crop(model, image_path, confidence, crop_size)
            processed_count += 1

            # Get ground truth class for this image
            ground_truth_class = get_ground_truth_class(image_path, input_dir)

            # Get class name for directory structure
            if 0 <= ground_truth_class < len(class_names):
                class_name = class_names[ground_truth_class]
            else:
                class_name = class_names[0]  # Default to first class

            for i, crop_data in enumerate(crops):
                # Generate crop filename
                crop_filename = f"{image_path.stem}_crop_{i:03d}.jpg"

                # Determine output path based on split and class (PyTorch ImageFolder structure)
                if split == 'all':
                    # If no split structure, save to default class folder
                    class_dir = crops_dir / class_name
                    class_dir.mkdir(exist_ok=True)
                    crop_output_path = class_dir / crop_filename
                else:
                    # Save to split/class/filename.jpg structure
                    crop_output_path = crops_dir / split / class_name / crop_filename

                # Save crop
                cv2.imwrite(str(crop_output_path), crop_data['crop'])

                # Add metadata
                metadata.append({
                    'original_image': str(image_path.relative_to(input_path)),
                    'crop_filename': crop_filename,
                    'split': split,
                    'confidence': crop_data['confidence'],
                    'bbox_x1': crop_data['bbox'][0],
                    'bbox_y1': crop_data['bbox'][1],
                    'bbox_x2': crop_data['bbox'][2],
                    'bbox_y2': crop_data['bbox'][3],
                    'crop_x1': crop_data['crop_coords'][0],
                    'crop_y1': crop_data['crop_coords'][1],
                    'crop_x2': crop_data['crop_coords'][2],
                    'crop_y2': crop_data['crop_coords'][3],
                    'dataset_source': dataset_name,
                    'ground_truth_class': ground_truth_class
                })

                crop_count += 1

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue

    # Save metadata
    if metadata:
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(output_path / 'crop_metadata.csv', index=False)

        print(f"\n✅ Processing completed:")
        print(f"   📸 Images processed: {processed_count}")
        print(f"   ✂️  Crops generated: {crop_count}")
        print(f"   📊 Average crops per image: {crop_count/processed_count:.2f}")

        # Show split distribution
        if 'split' in metadata_df.columns:
            split_counts = metadata_df['split'].value_counts()
            print(f"   📂 Split distribution:")
            for split, count in split_counts.items():
                print(f"      {split}: {count} crops")

        # Show class distribution
        if 'ground_truth_class' in metadata_df.columns:
            class_counts = metadata_df['ground_truth_class'].value_counts().sort_index()
            print(f"   🏷️  Class distribution:")
            for class_id, count in class_counts.items():
                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                    print(f"      {class_name} (class {class_id}): {count} crops")

        return metadata_df
    else:
        print("❌ No crops were generated!")
        return None

def create_yolo_classification_structure(crops_dir, metadata_df, output_dir):
    """Create YOLO classification directory structure"""
    yolo_dir = Path(output_dir) / "yolo_classification"

    # For now, create single class structure (all crops are "parasite")
    # Later this can be extended to use species labels if available

    for split in ['train', 'val', 'test']:
        split_crops = metadata_df[metadata_df['split'] == split]
        if len(split_crops) > 0:
            # Create parasite class directory
            class_dir = yolo_dir / split / "parasite"
            class_dir.mkdir(parents=True, exist_ok=True)

            # Copy crops to class directory
            for _, row in split_crops.iterrows():
                src_path = Path(crops_dir) / split / row['crop_filename']
                dst_path = class_dir / row['crop_filename']

                if src_path.exists():
                    import shutil
                    shutil.copy2(src_path, dst_path)

    print(f"✅ YOLO classification structure created at: {yolo_dir}")
    return yolo_dir

def main():
    parser = argparse.ArgumentParser(description="Generate crops from detection model")
    parser.add_argument("--model", required=True,
                       help="Path to trained detection model (best.pt)")
    parser.add_argument("--input", required=True,
                       help="Input dataset directory (with train/val/test/images structure)")
    parser.add_argument("--output", required=True,
                       help="Output directory for generated crops")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Detection confidence threshold")
    parser.add_argument("--crop_size", type=int, default=128,
                       help="Size of generated crops")
    parser.add_argument("--dataset_name", default="multispecies",
                       help="Name of source dataset for metadata")
    parser.add_argument("--create_yolo_structure", action="store_true",
                       help="Create YOLO classification directory structure")
    parser.add_argument("--fix_classification_structure", action="store_true",
                       help="Fix classification structure to use 4 malaria species classes")

    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING CROPS FROM DETECTION MODEL")
    print("=" * 60)

    # Validate inputs
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return

    if not Path(args.input).exists():
        print(f"❌ Input directory not found: {args.input}")
        return

    print(f"🎯 Detection model: {args.model}")
    print(f"📁 Input dataset: {args.input}")
    print(f"📂 Output directory: {args.output}")
    print(f"🎚️  Confidence threshold: {args.confidence}")
    print(f"📏 Crop size: {args.crop_size}x{args.crop_size}")

    try:
        # Load detection model
        model = load_detection_model(args.model)

        # Process dataset
        metadata = process_dataset(
            model=model,
            input_dir=args.input,
            output_dir=args.output,
            dataset_name=args.dataset_name,
            confidence=args.confidence,
            crop_size=args.crop_size
        )

        if metadata is not None and args.create_yolo_structure:
            # Create YOLO classification structure
            crops_dir = Path(args.output) / "crops"
            yolo_dir = create_yolo_classification_structure(
                crops_dir, metadata, args.output
            )

            # Fix classification structure if requested
            if args.fix_classification_structure:
                print(f"\n🔧 Fixing classification structure for 4 malaria species...")
                fix_cmd = [
                    "python3", "fix_classification_structure.py",
                    "--crop_data_path", args.output,
                    "--input_path", args.input
                ]

                result = subprocess.run(fix_cmd, capture_output=False, text=True)
                if result.returncode == 0:
                    print(f"✅ Classification structure fixed successfully!")
                else:
                    print(f"❌ Failed to fix classification structure")

        print(f"\n🎉 Crop generation completed successfully!")
        print(f"📊 Results saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return

if __name__ == "__main__":
    main()