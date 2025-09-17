#!/usr/bin/env python3
"""
Annotation utility functions for malaria detection datasets
Author: Malaria Detection Team
Date: 2024
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import csv


def load_json_annotations(json_path: Path) -> List[Dict]:
    """
    Load annotations from JSON file
    
    Args:
        json_path: Path to JSON annotation file
        
    Returns:
        List of annotation dictionaries
    """
    
    try:
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        if isinstance(annotations, dict):
            # Handle different JSON structures
            if 'annotations' in annotations:
                return annotations['annotations']
            elif 'images' in annotations:
                return annotations['images']
            else:
                return [annotations]
        elif isinstance(annotations, list):
            return annotations
        else:
            return []
            
    except Exception as e:
        print(f"Error loading JSON annotations from {json_path}: {e}")
        return []


def save_json_annotations(annotations: List[Dict], 
                         output_path: Path,
                         indent: int = 2) -> bool:
    """
    Save annotations to JSON file
    
    Args:
        annotations: List of annotation dictionaries
        output_path: Path to save JSON file
        indent: JSON indentation
        
    Returns:
        bool: True if successful
    """
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=indent)
        
        return True
        
    except Exception as e:
        print(f"Error saving JSON annotations to {output_path}: {e}")
        return False


def load_csv_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Load annotations from CSV file
    
    Args:
        csv_path: Path to CSV annotation file
        
    Returns:
        DataFrame with annotations
    """
    
    try:
        df = pd.read_csv(csv_path)
        return df
        
    except Exception as e:
        print(f"Error loading CSV annotations from {csv_path}: {e}")
        return pd.DataFrame()


def save_csv_annotations(annotations: Union[List[Dict], pd.DataFrame], 
                        output_path: Path) -> bool:
    """
    Save annotations to CSV file
    
    Args:
        annotations: List of dictionaries or DataFrame
        output_path: Path to save CSV file
        
    Returns:
        bool: True if successful
    """
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(annotations, list):
            df = pd.DataFrame(annotations)
        else:
            df = annotations
        
        df.to_csv(output_path, index=False)
        return True
        
    except Exception as e:
        print(f"Error saving CSV annotations to {output_path}: {e}")
        return False


def load_yolo_annotations(label_path: Path, 
                         image_size: Optional[Tuple[int, int]] = None) -> List[Dict]:
    """
    Load YOLO format annotations
    
    Args:
        label_path: Path to YOLO .txt file
        image_size: Image size (width, height) for denormalization
        
    Returns:
        List of annotation dictionaries
    """
    
    annotations = []
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            if len(parts) >= 1:
                class_id = int(parts[0])
                
                annotation = {
                    'class_id': class_id,
                    'format': 'yolo'
                }
                
                # For detection format (class_id x_center y_center width height)
                if len(parts) == 5:
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    annotation.update({
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'normalized': True
                    })
                    
                    # Convert to absolute coordinates if image size provided
                    if image_size:
                        img_w, img_h = image_size
                        
                        abs_width = width * img_w
                        abs_height = height * img_h
                        abs_x_center = x_center * img_w
                        abs_y_center = y_center * img_h
                        
                        # Calculate bounding box coordinates
                        x1 = abs_x_center - abs_width / 2
                        y1 = abs_y_center - abs_height / 2
                        x2 = abs_x_center + abs_width / 2
                        y2 = abs_y_center + abs_height / 2
                        
                        annotation.update({
                            'bbox': [x1, y1, x2, y2],
                            'area': abs_width * abs_height
                        })
                
                annotations.append(annotation)
                
    except Exception as e:
        print(f"Error loading YOLO annotations from {label_path}: {e}")
    
    return annotations


def save_yolo_annotations(annotations: List[Dict], 
                         output_path: Path,
                         image_size: Optional[Tuple[int, int]] = None) -> bool:
    """
    Save annotations in YOLO format
    
    Args:
        annotations: List of annotation dictionaries
        output_path: Path to save .txt file
        image_size: Image size for normalization if needed
        
    Returns:
        bool: True if successful
    """
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for ann in annotations:
                class_id = ann.get('class_id', 0)
                
                # For classification (just class ID)
                if 'bbox' not in ann and 'x_center' not in ann:
                    f.write(f"{class_id}\\n")
                
                # For detection with normalized coordinates
                elif 'x_center' in ann and ann.get('normalized', True):
                    x_center = ann['x_center']
                    y_center = ann['y_center']
                    width = ann['width']
                    height = ann['height']
                    
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\\n")
                
                # For detection with absolute coordinates - need to normalize
                elif 'bbox' in ann and image_size:
                    x1, y1, x2, y2 = ann['bbox']
                    img_w, img_h = image_size
                    
                    # Calculate normalized coordinates
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\\n")
        
        return True
        
    except Exception as e:
        print(f"Error saving YOLO annotations to {output_path}: {e}")
        return False


def load_xml_annotations(xml_path: Path) -> List[Dict]:
    """
    Load Pascal VOC XML format annotations
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        List of annotation dictionaries
    """
    
    annotations = []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image information
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            width = height = None
        
        filename = root.find('filename')
        filename = filename.text if filename is not None else None
        
        # Get objects
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                annotation = {
                    'filename': filename,
                    'class_name': class_name,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'area': (xmax - xmin) * (ymax - ymin),
                    'image_width': width,
                    'image_height': height,
                    'format': 'pascal_voc'
                }
                
                annotations.append(annotation)
                
    except Exception as e:
        print(f"Error loading XML annotations from {xml_path}: {e}")
    
    return annotations


def convert_annotations_format(annotations: List[Dict],
                             source_format: str,
                             target_format: str,
                             class_mapping: Optional[Dict] = None,
                             image_size: Optional[Tuple[int, int]] = None) -> List[Dict]:
    """
    Convert annotations between different formats
    
    Args:
        annotations: List of annotation dictionaries
        source_format: Source format ('yolo', 'pascal_voc', 'coco', etc.)
        target_format: Target format
        class_mapping: Mapping from class names to IDs
        image_size: Image size for coordinate conversion
        
    Returns:
        List of converted annotations
    """
    
    converted = []
    
    for ann in annotations:
        converted_ann = ann.copy()
        
        # Convert class names to IDs if needed
        if 'class_name' in ann and class_mapping and target_format == 'yolo':
            class_name = ann['class_name']
            class_id = class_mapping.get(class_name, 0)
            converted_ann['class_id'] = class_id
        
        # Convert between coordinate formats
        if source_format == 'pascal_voc' and target_format == 'yolo':
            if 'bbox' in ann and image_size:
                x1, y1, x2, y2 = ann['bbox']
                img_w, img_h = image_size
                
                # Convert to YOLO format
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                
                converted_ann.update({
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'normalized': True,
                    'format': 'yolo'
                })
        
        elif source_format == 'yolo' and target_format == 'pascal_voc':
            if 'x_center' in ann and image_size:
                x_center = ann['x_center']
                y_center = ann['y_center']
                width = ann['width']
                height = ann['height']
                img_w, img_h = image_size
                
                # Convert to Pascal VOC format
                abs_width = width * img_w
                abs_height = height * img_h
                abs_x_center = x_center * img_w
                abs_y_center = y_center * img_h
                
                x1 = abs_x_center - abs_width / 2
                y1 = abs_y_center - abs_height / 2
                x2 = abs_x_center + abs_width / 2
                y2 = abs_y_center + abs_height / 2
                
                converted_ann.update({
                    'bbox': [x1, y1, x2, y2],
                    'area': abs_width * abs_height,
                    'format': 'pascal_voc'
                })
        
        converted.append(converted_ann)
    
    return converted


def validate_annotations(annotations: List[Dict],
                        annotation_format: str = 'yolo',
                        image_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    Validate annotation format and content
    
    Args:
        annotations: List of annotation dictionaries
        annotation_format: Format to validate against
        image_size: Image size for coordinate validation
        
    Returns:
        Dictionary with validation results
    """
    
    validation_results = {
        'valid': True,
        'total_annotations': len(annotations),
        'errors': [],
        'warnings': [],
        'statistics': {
            'class_distribution': Counter(),
            'bbox_sizes': [],
            'coordinate_issues': 0
        }
    }
    
    for i, ann in enumerate(annotations):
        try:
            # Check required fields based on format
            if annotation_format == 'yolo':
                if 'class_id' not in ann:
                    validation_results['errors'].append(f"Annotation {i}: Missing class_id")
                    validation_results['valid'] = False
                
                # For detection format
                if 'x_center' in ann or 'y_center' in ann:
                    required_fields = ['x_center', 'y_center', 'width', 'height']
                    missing_fields = [f for f in required_fields if f not in ann]
                    
                    if missing_fields:
                        validation_results['errors'].append(f"Annotation {i}: Missing fields {missing_fields}")
                        validation_results['valid'] = False
                    else:
                        # Validate coordinate ranges
                        if not (0 <= ann['x_center'] <= 1 and 0 <= ann['y_center'] <= 1):
                            validation_results['warnings'].append(f"Annotation {i}: Center coordinates outside [0,1]")
                        
                        if not (0 < ann['width'] <= 1 and 0 < ann['height'] <= 1):
                            validation_results['warnings'].append(f"Annotation {i}: Width/height outside (0,1]")
                        
                        # Check for very small bounding boxes
                        if ann['width'] * ann['height'] < 0.0001:  # Very small area
                            validation_results['warnings'].append(f"Annotation {i}: Very small bounding box")
                        
                        validation_results['statistics']['bbox_sizes'].append(ann['width'] * ann['height'])
            
            elif annotation_format == 'pascal_voc':
                required_fields = ['class_name', 'bbox']
                missing_fields = [f for f in required_fields if f not in ann]
                
                if missing_fields:
                    validation_results['errors'].append(f"Annotation {i}: Missing fields {missing_fields}")
                    validation_results['valid'] = False
                
                if 'bbox' in ann:
                    x1, y1, x2, y2 = ann['bbox']
                    
                    # Check coordinate order
                    if x2 <= x1 or y2 <= y1:
                        validation_results['errors'].append(f"Annotation {i}: Invalid bbox coordinates")
                        validation_results['valid'] = False
                    
                    # Check against image size if provided
                    if image_size:
                        img_w, img_h = image_size
                        if x2 > img_w or y2 > img_h or x1 < 0 or y1 < 0:
                            validation_results['warnings'].append(f"Annotation {i}: Bbox outside image bounds")
                    
                    validation_results['statistics']['bbox_sizes'].append((x2-x1) * (y2-y1))
            
            # Update class distribution
            if 'class_id' in ann:
                validation_results['statistics']['class_distribution'][ann['class_id']] += 1
            elif 'class_name' in ann:
                validation_results['statistics']['class_distribution'][ann['class_name']] += 1
            
        except Exception as e:
            validation_results['errors'].append(f"Annotation {i}: Error during validation - {str(e)}")
            validation_results['valid'] = False
    
    # Calculate summary statistics
    if validation_results['statistics']['bbox_sizes']:
        bbox_sizes = validation_results['statistics']['bbox_sizes']
        validation_results['statistics']['bbox_stats'] = {
            'mean_area': np.mean(bbox_sizes),
            'std_area': np.std(bbox_sizes),
            'min_area': min(bbox_sizes),
            'max_area': max(bbox_sizes)
        }
    
    return validation_results


def merge_annotation_files(annotation_paths: List[Path],
                          output_path: Path,
                          format_type: str = 'csv') -> bool:
    """
    Merge multiple annotation files
    
    Args:
        annotation_paths: List of paths to annotation files
        output_path: Path to save merged annotations
        format_type: Output format ('csv', 'json')
        
    Returns:
        bool: True if successful
    """
    
    try:
        all_annotations = []
        
        for path in annotation_paths:
            if path.suffix.lower() == '.csv':
                df = load_csv_annotations(path)
                annotations = df.to_dict('records')
            elif path.suffix.lower() == '.json':
                annotations = load_json_annotations(path)
            else:
                print(f"Unsupported annotation format: {path}")
                continue
            
            # Add source information
            for ann in annotations:
                ann['source_file'] = str(path)
            
            all_annotations.extend(annotations)
        
        # Save merged annotations
        if format_type == 'csv':
            return save_csv_annotations(all_annotations, output_path)
        elif format_type == 'json':
            return save_json_annotations(all_annotations, output_path)
        else:
            print(f"Unsupported output format: {format_type}")
            return False
            
    except Exception as e:
        print(f"Error merging annotation files: {e}")
        return False


def create_class_mapping(annotations: List[Dict],
                        class_name_key: str = 'class_name') -> Dict[str, int]:
    """
    Create mapping from class names to class IDs
    
    Args:
        annotations: List of annotation dictionaries
        class_name_key: Key for class name in annotations
        
    Returns:
        Dictionary mapping class names to IDs
    """
    
    class_names = set()
    
    for ann in annotations:
        if class_name_key in ann:
            class_names.add(ann[class_name_key])
    
    # Create mapping with sorted class names for consistency
    class_mapping = {name: idx for idx, name in enumerate(sorted(class_names))}
    
    return class_mapping


def analyze_annotation_distribution(annotations: List[Dict]) -> Dict[str, Any]:
    """
    Analyze distribution of annotations
    
    Args:
        annotations: List of annotation dictionaries
        
    Returns:
        Dictionary with distribution analysis
    """
    
    analysis = {
        'total_annotations': len(annotations),
        'class_distribution': Counter(),
        'bbox_statistics': {},
        'format_distribution': Counter()
    }
    
    bbox_areas = []
    bbox_aspect_ratios = []
    
    for ann in annotations:
        # Class distribution
        if 'class_id' in ann:
            analysis['class_distribution'][ann['class_id']] += 1
        elif 'class_name' in ann:
            analysis['class_distribution'][ann['class_name']] += 1
        
        # Format distribution
        if 'format' in ann:
            analysis['format_distribution'][ann['format']] += 1
        
        # Bounding box statistics
        if 'bbox' in ann:
            x1, y1, x2, y2 = ann['bbox']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            bbox_areas.append(area)
            bbox_aspect_ratios.append(aspect_ratio)
        
        elif 'width' in ann and 'height' in ann:
            area = ann['width'] * ann['height']
            aspect_ratio = ann['width'] / ann['height'] if ann['height'] > 0 else 0
            
            bbox_areas.append(area)
            bbox_aspect_ratios.append(aspect_ratio)
    
    # Calculate bbox statistics
    if bbox_areas:
        analysis['bbox_statistics'] = {
            'areas': {
                'mean': np.mean(bbox_areas),
                'std': np.std(bbox_areas),
                'min': min(bbox_areas),
                'max': max(bbox_areas),
                'median': np.median(bbox_areas)
            },
            'aspect_ratios': {
                'mean': np.mean(bbox_aspect_ratios),
                'std': np.std(bbox_aspect_ratios),
                'min': min(bbox_aspect_ratios),
                'max': max(bbox_aspect_ratios),
                'median': np.median(bbox_aspect_ratios)
            }
        }
    
    return analysis