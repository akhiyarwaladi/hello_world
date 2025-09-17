#!/usr/bin/env python3
"""
Image processing utility functions for malaria detection
Author: Malaria Detection Team
Date: 2024
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def load_image(image_path: Path, color_mode: str = 'BGR') -> Optional[np.ndarray]:
    """
    Load image from path
    
    Args:
        image_path: Path to image file
        color_mode: Color mode ('BGR', 'RGB', 'GRAY')
        
    Returns:
        Loaded image array or None if failed
    """
    
    try:
        if color_mode == 'BGR':
            img = cv2.imread(str(image_path))
        elif color_mode == 'RGB':
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_mode == 'GRAY':
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(f"Unsupported color mode: {color_mode}")
        
        return img
        
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, 
              output_path: Path, 
              quality: int = 95) -> bool:
    """
    Save image to file
    
    Args:
        image: Image array
        output_path: Path to save image
        quality: JPEG quality (0-100)
        
    Returns:
        bool: True if successful
    """
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.jpg' or output_path.suffix.lower() == '.jpeg':
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_path.suffix.lower() == '.png':
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(str(output_path), image)
        
        return True
        
    except Exception as e:
        print(f"Failed to save image {output_path}: {e}")
        return False


def resize_image(image: np.ndarray, 
                target_size: Tuple[int, int],
                maintain_aspect: bool = True,
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    
    if not maintain_aspect:
        return cv2.resize(image, target_size, interpolation=interpolation)
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    return resized


def pad_image_to_square(image: np.ndarray, 
                       target_size: int,
                       pad_color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    """
    Pad image to square with specified color
    
    Args:
        image: Input image
        target_size: Target square size
        pad_color: RGB color for padding
        
    Returns:
        Padded square image
    """
    
    h, w = image.shape[:2]
    
    # Calculate padding
    pad_h = target_size - h
    pad_w = target_size - w
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # Pad image
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=pad_color
    )
    
    return padded


def resize_and_pad(image: np.ndarray, 
                  target_size: int,
                  pad_color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    """
    Resize image maintaining aspect ratio and pad to square
    
    Args:
        image: Input image
        target_size: Target square size
        pad_color: RGB color for padding
        
    Returns:
        Resized and padded image
    """
    
    # Resize maintaining aspect ratio
    resized = resize_image(image, (target_size, target_size), maintain_aspect=True)
    
    # Pad to square
    padded = pad_image_to_square(resized, target_size, pad_color)
    
    return padded


def enhance_image_quality(image: np.ndarray, 
                         enhance_contrast: bool = True,
                         enhance_sharpness: bool = True,
                         denoise: bool = True) -> np.ndarray:
    """
    Enhance image quality
    
    Args:
        image: Input image
        enhance_contrast: Whether to enhance contrast
        enhance_sharpness: Whether to enhance sharpness
        denoise: Whether to apply denoising
        
    Returns:
        Enhanced image
    """
    
    enhanced = image.copy()
    
    # Convert to LAB color space for better contrast enhancement
    if len(enhanced.shape) == 3:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
        
        # Merge channels and convert back
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Denoising
        if denoise:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
    else:
        # Grayscale image
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
        
        if denoise:
            enhanced = cv2.fastNlMeansDenoising(enhanced)
    
    # Sharpening
    if enhance_sharpness:
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced


def calculate_image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate image quality metrics
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of quality metrics
    """
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    metrics = {}
    
    # Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    metrics['sharpness'] = float(laplacian_var)
    
    # Contrast (standard deviation)
    metrics['contrast'] = float(np.std(gray))
    
    # Brightness (mean intensity)
    metrics['brightness'] = float(np.mean(gray))
    
    # Dynamic range
    metrics['dynamic_range'] = float(np.max(gray) - np.min(gray))
    
    # Entropy (information content)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))
    metrics['entropy'] = float(entropy)
    
    # Signal-to-noise ratio estimation
    # Simple estimation using mean/std
    if np.std(gray) > 0:
        snr = np.mean(gray) / np.std(gray)
        metrics['snr_estimate'] = float(snr)
    else:
        metrics['snr_estimate'] = 0.0
    
    return metrics


def detect_edges(image: np.ndarray, 
                low_threshold: int = 50,
                high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges using Canny edge detector
    
    Args:
        image: Input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection
        
    Returns:
        Edge image
    """
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges


def find_contours_and_features(image: np.ndarray, 
                              min_area: int = 100) -> List[Dict[str, Any]]:
    """
    Find contours and extract features
    
    Args:
        image: Input binary/edge image
        min_area: Minimum contour area
        
    Returns:
        List of contour features
    """
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area >= min_area:
            # Calculate features
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Extent (contour area / bounding rectangle area)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            # Solidity (contour area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            features.append({
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'bounding_box': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity,
                'circularity': circularity
            })
    
    return features


def create_image_grid(images: List[np.ndarray], 
                     titles: Optional[List[str]] = None,
                     grid_size: Optional[Tuple[int, int]] = None,
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a grid of images for visualization
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        grid_size: Grid size (rows, cols). If None, auto-calculate
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    
    n_images = len(images)
    
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < n_images:
            img = images[i]
            
            # Convert BGR to RGB for display if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Assume BGR, convert to RGB
                img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_display = img
            
            ax.imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
            
            if titles and i < len(titles):
                ax.set_title(titles[i])
            
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_confusion_matrix_plot(y_true: List, 
                                y_pred: List,
                                class_names: List[str],
                                figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Create confusion matrix visualization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig


def batch_process_images(image_paths: List[Path],
                        output_dir: Path,
                        processing_func: callable,
                        **kwargs) -> List[Path]:
    """
    Process multiple images in batch
    
    Args:
        image_paths: List of input image paths
        output_dir: Output directory
        processing_func: Function to apply to each image
        **kwargs: Additional arguments for processing function
        
    Returns:
        List of output image paths
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    
    from tqdm import tqdm
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Load image
        img = load_image(img_path)
        
        if img is not None:
            # Apply processing
            processed_img = processing_func(img, **kwargs)
            
            # Save processed image
            output_path = output_dir / img_path.name
            
            if save_image(processed_img, output_path):
                output_paths.append(output_path)
    
    return output_paths


def calculate_dataset_statistics(image_dir: Path) -> Dict[str, Any]:
    """
    Calculate statistics for image dataset
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        Dictionary of dataset statistics
    """
    
    stats = {
        'total_images': 0,
        'resolutions': [],
        'file_sizes': [],
        'quality_metrics': {
            'sharpness': [],
            'contrast': [],
            'brightness': []
        }
    }
    
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    from tqdm import tqdm
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(list(image_dir.glob(f"**/*{ext}")))
        image_files.extend(list(image_dir.glob(f"**/*{ext.upper()}")))
    
    for img_path in tqdm(image_files, desc="Analyzing images"):
        try:
            # Load image
            img = load_image(img_path)
            
            if img is not None:
                # Basic stats
                stats['total_images'] += 1
                h, w = img.shape[:2]
                stats['resolutions'].append((w, h))
                stats['file_sizes'].append(img_path.stat().st_size)
                
                # Quality metrics
                quality = calculate_image_quality_metrics(img)
                stats['quality_metrics']['sharpness'].append(quality['sharpness'])
                stats['quality_metrics']['contrast'].append(quality['contrast'])
                stats['quality_metrics']['brightness'].append(quality['brightness'])
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate summary statistics
    if stats['resolutions']:
        widths = [r[0] for r in stats['resolutions']]
        heights = [r[1] for r in stats['resolutions']]
        
        stats['resolution_stats'] = {
            'mean_width': np.mean(widths),
            'mean_height': np.mean(heights),
            'min_width': min(widths),
            'max_width': max(widths),
            'min_height': min(heights),
            'max_height': max(heights)
        }
        
        stats['file_size_stats'] = {
            'mean_mb': np.mean(stats['file_sizes']) / (1024*1024),
            'total_mb': sum(stats['file_sizes']) / (1024*1024),
            'min_mb': min(stats['file_sizes']) / (1024*1024),
            'max_mb': max(stats['file_sizes']) / (1024*1024)
        }
        
        # Quality statistics
        for metric in ['sharpness', 'contrast', 'brightness']:
            values = stats['quality_metrics'][metric]
            if values:
                stats['quality_metrics'][f'{metric}_mean'] = np.mean(values)
                stats['quality_metrics'][f'{metric}_std'] = np.std(values)
                stats['quality_metrics'][f'{metric}_min'] = min(values)
                stats['quality_metrics'][f'{metric}_max'] = max(values)
    
    return stats