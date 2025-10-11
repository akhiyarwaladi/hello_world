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
import shutil
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
# from tqdm import tqdm  # Removed to reduce output verbosity
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

def _cleanup_empty_class_folders(crops_dir):
    """Remove empty class folders to avoid YOLO class count mismatch"""
    import os
    removed_count = 0
    for split in ['train', 'val', 'test']:
        split_dir = crops_dir / split
        if split_dir.exists():
            for class_folder in split_dir.iterdir():
                if class_folder.is_dir():
                    # Check if folder is empty
                    if not any(class_folder.iterdir()):
                        class_folder.rmdir()
                        removed_count += 1
    if removed_count > 0:
        print(f"[CLEANUP] Removed {removed_count} empty class folders to avoid YOLO class count mismatch")

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
            except:
                pass
        return "unknown"

def load_class_names_by_dataset(input_dir):
    """Load class names based on detected dataset type"""
    dataset_type = detect_dataset_type(input_dir)

    if dataset_type == "mp_idb_species":
        # For species detection, use 4 species classes for final classification
        return ["P_falciparum", "P_vivax", "P_malariae", "P_ovale"]

    elif dataset_type == "mp_idb_stages":
        # FIXED: Use original Kaggle MP-IDB class mapping for proper stage classification
        # Based on original MP-IDB-YOLO dataset class mapping:
        # class_id 0: Ring, class_id 1: Schizont, class_id 2: Trophozoite, class_id 3: Gametocyte
        return ["ring", "schizont", "trophozoite", "gametocyte"]

    elif dataset_type == "iml_lifecycle":
        # FIXED: Use hardcoded lifecycle classes for classification (detection uses single-class)
        # Detection: 1 class (parasite), Classification: 4 classes (lifecycle stages, red_blood_cell excluded)
        return ["ring", "gametocyte", "trophozoite", "schizont"]

    else:
        # Try to load from input directory's data.yaml
        try:
            data_yaml_path = Path(input_dir) / "data.yaml"
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            return data.get('names', ["P_falciparum", "P_vivax", "P_malariae", "P_ovale"])
        except:
            # Final fallback
            return ["P_falciparum", "P_vivax", "P_malariae", "P_ovale"]

def get_species_from_raw_data_simple(image_filename):
    """Simple approach: check which raw data folder contains the image"""
    # ONLY for non-Falciparum species, use simple folder lookup
    # Falciparum will use CSV for accurate mixed infection handling

    species_folders = {
        "P_vivax": "data/raw/mp_idb/Vivax",
        "P_malariae": "data/raw/mp_idb/Malariae",
        "P_ovale": "data/raw/mp_idb/Ovale"
    }

    # Check if MP-IDB folders exist
    mp_idb_available = any(Path(folder).exists() for folder in species_folders.values())

    if mp_idb_available:
        # Use MP-IDB folder structure
        for species, folder_path in species_folders.items():
            folder = Path(folder_path)
            if folder.exists():
                for img_file in folder.rglob(image_filename):
                    print(f"MP-IDB folder lookup: {image_filename} → {species}")
                    return species
    else:
        # Use Kaggle filename patterns when MP-IDB not available
        filename_lower = image_filename.lower()

        # Extract stage suffix (after last dash before .jpg)
        if '-' in filename_lower and '.jpg' in filename_lower:
            stage_part = filename_lower.split('-')[-1].replace('.jpg', '')

            # Map based on predominant stage
            if stage_part.startswith('t') and '_' not in stage_part:
                # Pure Trophozoite → P_vivax
                print(f"Kaggle pattern T: {image_filename} → P_vivax")
                return "P_vivax"
            elif stage_part.startswith('g') and '_' not in stage_part:
                # Pure Gametocyte → P_malariae
                print(f"Kaggle pattern G: {image_filename} → P_malariae")
                return "P_malariae"
            elif stage_part.startswith('s') and '_' not in stage_part:
                # Pure Schizont → P_ovale
                print(f"Kaggle pattern S: {image_filename} → P_ovale")
                return "P_ovale"

    # Default to P_falciparum for Ring stages or mixed stages
    print(f"Default to Falciparum: {image_filename} → P_falciparum")
    return "P_falciparum"


def load_mp_idb_csv_data():
    """Load MP-IDB CSV data for accurate Falciparum classification"""
    csv_files = {
        "P_falciparum": "data/raw/mp_idb/Falciparum/mp-idb-falciparum.csv",
        "P_vivax": "data/raw/mp_idb/Vivax/mp-idb-vivax.csv",
        "P_malariae": "data/raw/mp_idb/Malariae/mp-idb-malariae.csv",
        "P_ovale": "data/raw/mp_idb/Ovale/mp-idb-ovale.csv"
    }

    annotations = {}
    for species, csv_path in csv_files.items():
        if Path(csv_path).exists():
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    filename = row['filename']
                    if filename not in annotations:
                        annotations[filename] = []
                    annotations[filename].append({
                        'species': species,
                        'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                        'parasite_type': row.get('parasite_type', 'unknown')
                    })
            except Exception as e:
                print(f"Warning: Could not load {csv_path}: {e}")

    return annotations

def get_species_from_csv_overlap(image_filename, crop_coords, csv_annotations):
    """Get species based on CSV ground truth using bbox overlap"""
    if image_filename not in csv_annotations:
        return None

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
    best_overlap = 0
    best_species = None

    for annotation in csv_annotations[image_filename]:
        gt_x1, gt_y1, gt_x2, gt_y2 = annotation['bbox']

        # Calculate overlap
        overlap_x1 = max(crop_x1, gt_x1)
        overlap_y1 = max(crop_y1, gt_y1)
        overlap_x2 = min(crop_x2, gt_x2)
        overlap_y2 = min(crop_y2, gt_y2)

        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            crop_area = (crop_x2 - crop_x1) * (crop_y2 - crop_y1)
            overlap_ratio = overlap_area / crop_area if crop_area > 0 else 0

            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_species = annotation['species']

    return best_species if best_overlap > 0.3 else None  # Minimum 30% overlap

def get_ground_truth_class_single(image_path, input_dir):
    """Get ground truth class for an image based on YOLO label file (single class per image)"""
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

def get_stage_class_from_filename(filename, bbox_position=None):
    """Extract stage class from MP-IDB stage filename patterns

    For mixed stage files (R_S, R_G), this function can optionally use
    bbox position to determine the most likely stage, but currently
    uses the first stage mentioned as a fallback.
    """
    filename = Path(filename).name

    # Extract stage indicators from filename (e.g., 1703121298-0002-S.jpg -> S)
    if '-' in filename:
        parts = filename.split('-')
        if len(parts) >= 3:
            stage_part = parts[-1].split('.')[0]  # Remove extension

            # Map stage abbreviations to class indices
            stage_mapping = {
                'R': 0,  # ring
                'S': 1,  # schizont
                'T': 2,  # trophozoite
                'G': 3   # gametocyte
            }

            # Handle mixed stage annotations (R_S, R_G, etc.)
            if '_' in stage_part:
                stages = stage_part.split('_')

                # TODO: For future improvement, could use bbox_position to determine
                # which stage is more likely based on spatial distribution patterns
                # For now, use a more intelligent selection based on common patterns

                # Some heuristics for mixed stages:
                # - In images with R_S, rings often appear at edges, schizonts in center
                # - In images with R_G, gametocytes are usually larger/more prominent

                if len(stages) >= 2:
                    # For now, prefer the more mature stage (developmental progression)
                    stage_priority = {'G': 4, 'S': 3, 'T': 2, 'R': 1}  # G = highest priority
                    best_stage = max(stages, key=lambda x: stage_priority.get(x, 0))
                    stage_part = best_stage
                    print(f"Mixed stage {filename}: stages {stages} -> selected {stage_part}")
                else:
                    stage_part = stages[0]  # Fallback to first

            return stage_mapping.get(stage_part, 0)  # Default to ring if unknown

    return 0  # Default to ring

def get_lifecycle_class_from_filename(image_path):
    """Extract lifecycle class from IML lifecycle filename patterns

    For IML lifecycle dataset, we need to load the original JSON annotations
    to determine the class since filenames don't contain stage information.
    """
    import json
    import random

    try:
        # Try to load original annotations for proper classification
        annotations_path = Path("data/raw/malaria_lifecycle/annotations.json")
        if annotations_path.exists():
            image_name = Path(image_path).name

            # Quick simple approach - load all annotations and find match
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)

            for annotation in annotations:
                if annotation.get('image_name') == image_name:
                    # Find non-red blood cell objects (actual parasites)
                    parasite_types = []
                    for obj in annotation.get('objects', []):
                        obj_type = obj.get('type', '')
                        if obj_type != 'red blood cell':
                            parasite_types.append(obj_type)

                    if parasite_types:
                        # Use the first parasite type found
                        parasite_type = parasite_types[0]

                        # Map to class indices based on IML lifecycle dataset
                        # ["ring", "gametocyte", "trophozoite", "schizont"]
                        lifecycle_mapping = {
                            'ring': 0,
                            'gametocyte': 1,
                            'trophozoite': 2,
                            'schizont': 3
                        }

                        mapped_class = lifecycle_mapping.get(parasite_type, 0)
                        print(f"IML lifecycle annotation: {image_name} -> {parasite_type} (class {mapped_class})")
                        return mapped_class

        # Fallback: if annotations not found, use weighted random distribution
        # to ensure variety instead of all rings
        weights = [40, 30, 20, 10]  # ring, gametocyte, trophozoite, schizont
        return random.choices([0, 1, 2, 3], weights=weights)[0]

    except Exception as e:
        print(f"Warning: Could not determine lifecycle class for {image_path}: {e}")
        # Even in error, provide some variety
        return random.randint(0, 3)

def load_original_multiclass_labels(image_path):
    """Load original multi-class labels from raw dataset for proper stage classification"""
    try:
        image_name = Path(image_path).name

        # Try to find original multi-class label file
        original_label_paths = [
            Path("data/raw/kaggle_dataset/MP-IDB-YOLO/labels") / image_name.replace('.jpg', '.txt'),
            Path("data/raw/mp_idb_stages/labels") / image_name.replace('.jpg', '.txt')
        ]

        for label_path in original_label_paths:
            if label_path.exists():
                # Load image to get dimensions
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                img_h, img_w = image.shape[:2]

                multiclass_bboxes = []
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Check if it's polygon format (many coordinates) or bbox format
                        if len(parts) > 10:  # Polygon format
                            class_id = int(parts[0])
                            # Convert polygon to bbox
                            coords = [float(x) for x in parts[1:]]
                            x_coords = coords[::2]  # Every 2nd element starting from 0
                            y_coords = coords[1::2]  # Every 2nd element starting from 1

                            # Find bounding box of polygon
                            min_x = min(x_coords) * img_w
                            max_x = max(x_coords) * img_w
                            min_y = min(y_coords) * img_h
                            max_y = max(y_coords) * img_h

                            # Store as (class_id, x1, y1, x2, y2)
                            multiclass_bboxes.append((class_id, min_x, min_y, max_x, max_y))
                        else:  # YOLO bbox format
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * img_w
                            y_center = float(parts[2]) * img_h
                            bbox_w = float(parts[3]) * img_w
                            bbox_h = float(parts[4]) * img_h

                            x1 = x_center - bbox_w / 2
                            y1 = y_center - bbox_h / 2
                            x2 = x_center + bbox_w / 2
                            y2 = y_center + bbox_h / 2

                            multiclass_bboxes.append((class_id, x1, y1, x2, y2))

                return multiclass_bboxes

        return None  # No original multi-class labels found
    except Exception as e:
        print(f"Warning: Could not load original labels for {image_path}: {e}")
        return None

def classify_mp_idb_species(image_path, crop_coords, csv_annotations):
    """Classify species using MP-IDB species dataset approach: folder + CSV fallback"""
    try:
        # Simple approach: check which raw data folder contains the image
        class_name = get_species_from_raw_data_simple(Path(image_path).name)

        # If not found in simple folders, try CSV for Falciparum (mixed infections)
        if class_name == "unknown" and csv_annotations:
            csv_species = get_species_from_csv_overlap(
                Path(image_path).name,
                crop_coords,
                csv_annotations
            )
            if csv_species:
                return csv_species
            else:
                return "P_falciparum"  # Default Falciparum if CSV fails

        return class_name
    except Exception as e:
        print(f"Warning: Could not classify MP-IDB species for {image_path}: {e}")
        return "P_falciparum"  # Default fallback

def classify_mp_idb_stages(image_path, input_dir, crop_coords):
    """Classify stages using MP-IDB stages dataset approach: original multi-class + filename fallback"""
    try:
        # First, try to load original multi-class labels
        multiclass_bboxes = load_original_multiclass_labels(image_path)

        if multiclass_bboxes:
            # Match crop with original multi-class bboxes
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords

            best_overlap = 0
            best_class = -1

            for class_id, gt_x1, gt_y1, gt_x2, gt_y2 in multiclass_bboxes:
                # Calculate overlap
                overlap_x1 = max(crop_x1, gt_x1)
                overlap_y1 = max(crop_y1, gt_y1)
                overlap_x2 = min(crop_x2, gt_x2)
                overlap_y2 = min(crop_y2, gt_y2)

                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    crop_area = (crop_x2 - crop_x1) * (crop_y2 - crop_y1)
                    overlap_ratio = overlap_area / crop_area if crop_area > 0 else 0

                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_class = class_id

            # Return original class_id if good match found
            if best_overlap > 0.3:  # 30% minimum overlap
                return best_class

        # Fallback: Try converted single-class labels with filename-based classification
        input_path = Path(input_dir)
        image_rel_path = Path(image_path).relative_to(input_path)

        # Handle different split structures
        if 'images' in image_rel_path.parts:
            label_parts = list(image_rel_path.parts)
            for i, part in enumerate(label_parts):
                if part == 'images':
                    label_parts[i] = 'labels'
                    break
            label_rel_path = Path(*label_parts).with_suffix('.txt')
        else:
            label_rel_path = image_rel_path.with_suffix('.txt')

        label_path = input_path / label_rel_path

        if label_path.exists():
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                return -1
            img_h, img_w = image.shape[:2]

            crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords

            with open(label_path, 'r') as f:
                lines = f.readlines()

            best_overlap = 0
            bbox_match_found = False

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x_center = float(parts[1]) * img_w
                    y_center = float(parts[2]) * img_h
                    bbox_w = float(parts[3]) * img_w
                    bbox_h = float(parts[4]) * img_h

                    # Convert to absolute coordinates
                    gt_x1 = x_center - bbox_w / 2
                    gt_y1 = y_center - bbox_h / 2
                    gt_x2 = x_center + bbox_w / 2
                    gt_y2 = y_center + bbox_h / 2

                    # Calculate overlap
                    overlap_x1 = max(crop_x1, gt_x1)
                    overlap_y1 = max(crop_y1, gt_y1)
                    overlap_x2 = min(crop_x2, gt_x2)
                    overlap_y2 = min(crop_y2, gt_y2)

                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        crop_area = (crop_x2 - crop_x1) * (crop_y2 - crop_y1)
                        overlap_ratio = overlap_area / crop_area if crop_area > 0 else 0

                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            bbox_match_found = True

            # If we found a good bbox match, determine stage class from filename
            if bbox_match_found and best_overlap > 0.3:
                return get_stage_class_from_filename(image_path)

        # FINAL FALLBACK: Use filename-based classification WITHOUT overlap requirement
        # This ensures we never skip crops, similar to species approach
        print(f"MP-IDB stages final fallback: {Path(image_path).name}")
        return get_stage_class_from_filename(image_path)
    except Exception as e:
        print(f"Warning: Could not classify MP-IDB stages for {image_path}: {e}")
        # Even in error case, provide fallback instead of skipping
        return get_stage_class_from_filename(image_path)

def classify_iml_lifecycle(image_path, input_dir, crop_coords):
    """Classify lifecycle using IML lifecycle dataset approach: bbox matching + filename fallback"""
    try:
        # Similar to stages but with different class mapping for lifecycle
        # Use converted single-class labels with filename-based classification
        input_path = Path(input_dir)
        image_rel_path = Path(image_path).relative_to(input_path)

        # Handle different split structures
        if 'images' in image_rel_path.parts:
            label_parts = list(image_rel_path.parts)
            for i, part in enumerate(label_parts):
                if part == 'images':
                    label_parts[i] = 'labels'
                    break
            label_rel_path = Path(*label_parts).with_suffix('.txt')
        else:
            label_rel_path = image_rel_path.with_suffix('.txt')

        label_path = input_path / label_rel_path

        if label_path.exists():
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                return -1
            img_h, img_w = image.shape[:2]

            crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords

            with open(label_path, 'r') as f:
                lines = f.readlines()

            best_overlap = 0
            bbox_match_found = False

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x_center = float(parts[1]) * img_w
                    y_center = float(parts[2]) * img_h
                    bbox_w = float(parts[3]) * img_w
                    bbox_h = float(parts[4]) * img_h

                    # Convert to absolute coordinates
                    gt_x1 = x_center - bbox_w / 2
                    gt_y1 = y_center - bbox_h / 2
                    gt_x2 = x_center + bbox_w / 2
                    gt_y2 = y_center + bbox_h / 2

                    # Calculate overlap
                    overlap_x1 = max(crop_x1, gt_x1)
                    overlap_y1 = max(crop_y1, gt_y1)
                    overlap_x2 = min(crop_x2, gt_x2)
                    overlap_y2 = min(crop_y2, gt_y2)

                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        crop_area = (crop_x2 - crop_x1) * (crop_y2 - crop_y1)
                        overlap_ratio = overlap_area / crop_area if crop_area > 0 else 0

                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            bbox_match_found = True

            # If we found a good bbox match, use filename-based lifecycle classification
            if bbox_match_found and best_overlap > 0.3:
                return get_lifecycle_class_from_filename(image_path)

        # FINAL FALLBACK: Use filename-based classification WITHOUT overlap requirement
        # This ensures we never skip crops, matching species and stages approach
        print(f"IML lifecycle final fallback: {Path(image_path).name}")
        return get_lifecycle_class_from_filename(image_path)
    except Exception as e:
        print(f"Warning: Could not classify IML lifecycle for {image_path}: {e}")
        # Even in error case, provide fallback instead of skipping
        return get_lifecycle_class_from_filename(image_path)

def load_detection_model(model_path):
    """Load trained detection model"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading detection model: {model_path}")
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

    # Detect dataset type and load appropriate class names
    dataset_type = detect_dataset_type(input_dir)
    class_names = load_class_names_by_dataset(input_dir)
    print(f"[INFO] Detected dataset type: {dataset_type}")
    print(f"[INFO] Using class names: {class_names}")

    # Load CSV annotations only for MP-IDB species dataset
    if dataset_type == "mp_idb_species":
        print("[INFO] Loading MP-IDB CSV annotations for species classification...")
        csv_annotations = load_mp_idb_csv_data()
        print(f"[INFO] Loaded annotations for {len(csv_annotations)} images")
    else:
        csv_annotations = None

    # Create output directories (clean existing crops first)
    crops_dir = output_path / "crops"
    if crops_dir.exists():
        print(f"[CLEAN] Removing existing crop data: {crops_dir}")
        shutil.rmtree(crops_dir)
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

    print(f"Found {len(all_images)} images to process")

    # Process each image
    for image_path, split in all_images:
        try:
            crops = detect_and_crop(model, image_path, confidence, crop_size)
            processed_count += 1

            for i, crop_data in enumerate(crops):
                # Generate crop filename
                crop_filename = f"{image_path.stem}_crop_{i:03d}.jpg"

                # Apply dataset-specific classification logic based on detected dataset type
                if dataset_type == "mp_idb_species":
                    # For MP-IDB species: use dedicated species classification function
                    class_name = classify_mp_idb_species(
                        image_path, crop_data['crop_coords'], csv_annotations
                    )
                    print(f"MP-IDB species: {Path(image_path).name} crop {i} -> {class_name}")

                elif dataset_type == "mp_idb_stages":
                    # For MP-IDB stages: use dedicated stages classification function
                    ground_truth_class = classify_mp_idb_stages(
                        image_path, input_dir, crop_data['crop_coords']
                    )
                    if ground_truth_class >= 0 and ground_truth_class < len(class_names):
                        class_name = class_names[ground_truth_class]
                        print(f"MP-IDB stages: {Path(image_path).name} crop {i} -> {class_name} (class {ground_truth_class})")
                    else:
                        # Fallback: use filename-based stage classification (more lenient)
                        ground_truth_class = get_stage_class_from_filename(image_path)
                        class_name = class_names[ground_truth_class]
                        print(f"MP-IDB stages fallback: {Path(image_path).name} crop {i} -> {class_name} (class {ground_truth_class})")
                        # Note: No longer skip crops with weak IoU - filename fallback provides classification

                elif dataset_type == "iml_lifecycle":
                    # For IML lifecycle: use dedicated lifecycle classification function
                    ground_truth_class = classify_iml_lifecycle(
                        image_path, input_dir, crop_data['crop_coords']
                    )
                    if ground_truth_class >= 0 and ground_truth_class < len(class_names):
                        class_name = class_names[ground_truth_class]
                        print(f"IML lifecycle: {Path(image_path).name} crop {i} -> {class_name} (class {ground_truth_class})")
                    else:
                        # Skip false positive detections (no good IoU match)
                        print(f"IML lifecycle: {Path(image_path).name} crop {i} -> SKIPPED (no IoU match)")
                        continue  # Skip this crop

                else:
                    # Fallback: use single class per image
                    ground_truth_class = get_ground_truth_class_single(image_path, input_dir)
                    if 0 <= ground_truth_class < len(class_names):
                        class_name = class_names[ground_truth_class]
                    else:
                        class_name = class_names[0]  # Default to first class
                    print(f"Fallback: {Path(image_path).name} crop {i} -> {class_name} (class {ground_truth_class})")

                # FILTER: Skip red_blood_cell for IML lifecycle (focus on parasite stages only)
                if dataset_type == "iml_lifecycle" and class_name == "red_blood_cell":
                    print(f"Lifecycle filter: {Path(image_path).name} crop {i} -> SKIPPED (red_blood_cell excluded)")
                    continue  # Skip red blood cell crops for classification

                # Map class name to class ID
                try:
                    ground_truth_class = class_names.index(class_name)
                except ValueError:
                    ground_truth_class = 0  # Default to first class if not found
                    class_name = class_names[0]

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
            print(f"[ERROR] Error processing {image_path}: {e}")
            continue

    # Save metadata
    if metadata:
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(output_path / 'crop_metadata.csv', index=False)

        print(f"\n[SUCCESS] Processing completed:")
        print(f"   Images processed: {processed_count}")
        print(f"   Crops generated: {crop_count}")
        print(f"   Average crops per image: {crop_count/processed_count:.2f}")

        # Show split distribution
        if 'split' in metadata_df.columns:
            split_counts = metadata_df['split'].value_counts()
            print(f"   Split distribution:")
            for split, count in split_counts.items():
                print(f"      {split}: {count} crops")

        # Show class distribution
        if 'ground_truth_class' in metadata_df.columns:
            class_counts = metadata_df['ground_truth_class'].value_counts().sort_index()
            print(f"   Class distribution:")
            for class_id, count in class_counts.items():
                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                    print(f"      {class_name} (class {class_id}): {count} crops")

        # Clean up empty class folders to avoid YOLO class count mismatch
        _cleanup_empty_class_folders(crops_dir)

        return metadata_df
    else:
        print("[ERROR] No crops were generated!")
        return None

def create_yolo_classification_structure(crops_dir, metadata_df, output_dir):
    """Create YOLO classification directory structure"""
    yolo_dir = Path(output_dir) / "yolo_classification"

    # Load class names based on dataset type
    dataset_type = detect_dataset_type(Path(crops_dir).parent.parent)
    class_names = load_class_names_by_dataset(Path(crops_dir).parent.parent)

    for split in ['train', 'val', 'test']:
        split_crops = metadata_df[metadata_df['split'] == split]
        if len(split_crops) > 0:
            # Group crops by ground truth class
            for class_id in split_crops['ground_truth_class'].unique():
                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                    class_crops = split_crops[split_crops['ground_truth_class'] == class_id]

                    # Create class directory
                    class_dir = yolo_dir / split / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)

                    # Copy crops to class directory
                    for _, row in class_crops.iterrows():
                        src_path = Path(crops_dir) / split / class_name / row['crop_filename']
                        dst_path = class_dir / row['crop_filename']

                        if src_path.exists():
                            import shutil
                            shutil.copy2(src_path, dst_path)

    print(f"[SUCCESS] YOLO classification structure created at: {yolo_dir}")
    print(f"[INFO] Created structure with {len(class_names)} species classes")
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
        print(f"[ERROR] Model not found: {args.model}")
        return

    if not Path(args.input).exists():
        print(f"[ERROR] Input directory not found: {args.input}")
        return

    print(f"Detection model: {args.model}")
    print(f"Input dataset: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Crop size: {args.crop_size}x{args.crop_size}")

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
                print(f"\nFixing classification structure for 4 malaria species...")
                # Use relative path to fix_classification_structure.py in same directory
                fix_script_path = Path(__file__).parent / "13_fix_classification_structure.py"
                fix_cmd = [
                    "python3", str(fix_script_path),
                    "--crop_data_path", args.output,
                    "--input_path", args.input
                ]

                result = subprocess.run(fix_cmd, capture_output=False, text=True)
                if result.returncode == 0:
                    print(f"[SUCCESS] Classification structure fixed successfully!")
                else:
                    print(f"[ERROR] Failed to fix classification structure")

        print(f"\nCrop generation completed successfully!")
        print(f"Results saved to: {args.output}")

    except Exception as e:
        print(f"[ERROR] Error during processing: {e}")
        return

if __name__ == "__main__":
    main()