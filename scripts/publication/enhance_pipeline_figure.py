"""
Enhanced Pipeline Architecture Figure for Q1 Journal
This script processes the pipeline architecture diagram to:
1. Remove excessive whitespace
2. Increase resolution/DPI to 300+ for publication quality
3. Optimize for journal submission
"""

import cv2
import numpy as np
from PIL import Image
import os

def remove_whitespace_crop(image_path, output_path, margin=20, threshold=250):
    """
    Crop whitespace from image while maintaining quality

    Args:
        image_path: Path to input image
        output_path: Path to save cropped image
        margin: Pixels to keep around content (default: 20)
        threshold: RGB threshold to consider as "white" (default: 250)
    """
    # Read image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    print(f"Original image shape: {img.shape}")

    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create binary mask: pixels darker than threshold are content
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Find all non-zero points (content)
    coords = cv2.findNonZero(binary)

    if coords is None:
        print("No content detected, saving original")
        cv2.imwrite(output_path, img)
        return

    # Get bounding box
    x, y, w, h = cv2.boundingRect(coords)

    print(f"Content bounding box: x={x}, y={y}, w={w}, h={h}")

    # Add margin but stay within image bounds
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(img.shape[1], x + w + margin)
    y_end = min(img.shape[0], y + h + margin)

    # Crop image
    cropped = img[y_start:y_end, x_start:x_end]

    print(f"Cropped image shape: {cropped.shape}")
    print(f"Size reduction: {img.shape[0]*img.shape[1]} -> {cropped.shape[0]*cropped.shape[1]} pixels")
    print(f"Reduction: {100*(1 - (cropped.shape[0]*cropped.shape[1])/(img.shape[0]*img.shape[1])):.1f}%")

    # Save cropped image
    cv2.imwrite(output_path, cropped)
    print(f"Saved cropped image to: {output_path}")

    return cropped


def enhance_for_publication(image_path, output_path, target_dpi=300, scale_factor=2.0):
    """
    Enhance image quality for publication

    Args:
        image_path: Path to input image
        output_path: Path to save enhanced image
        target_dpi: Target DPI for publication (default: 300)
        scale_factor: Upscaling factor for better quality (default: 2.0)
    """
    # Open with PIL for better DPI control
    img = Image.open(image_path)

    print(f"\n=== Enhancement Stage ===")
    print(f"Original size: {img.size}")
    print(f"Original mode: {img.mode}")

    # Get current DPI (if available)
    current_dpi = img.info.get('dpi', (72, 72))
    print(f"Current DPI: {current_dpi}")

    # Calculate new size for target DPI
    # If we want higher DPI, we need more pixels
    if scale_factor > 1.0:
        new_width = int(img.size[0] * scale_factor)
        new_height = int(img.size[1] * scale_factor)

        print(f"Upscaling to: {new_width}x{new_height}")

        # Use LANCZOS for high-quality upscaling
        img_upscaled = img.resize((new_width, new_height), Image.LANCZOS)
    else:
        img_upscaled = img

    # Save with high DPI
    img_upscaled.save(
        output_path,
        dpi=(target_dpi, target_dpi),
        quality=100,
        optimize=False
    )

    print(f"Saved enhanced image to: {output_path}")
    print(f"New size: {img_upscaled.size}")
    print(f"Target DPI: {target_dpi}")

    return img_upscaled


def create_publication_ready_figure(input_path, base_output_name="pipeline_architecture_enhanced"):
    """
    Complete pipeline to create publication-ready figure

    Args:
        input_path: Path to original image
        base_output_name: Base name for output files
    """
    output_dir = os.path.dirname(input_path)

    print("="*70)
    print("PIPELINE ARCHITECTURE FIGURE ENHANCEMENT FOR Q1 JOURNAL")
    print("="*70)

    # Step 1: Crop whitespace
    print("\n[STEP 1] Removing whitespace...")
    cropped_path = os.path.join(output_dir, f"{base_output_name}_cropped.png")
    cropped_img = remove_whitespace_crop(input_path, cropped_path, margin=30)

    # Step 2: Enhance quality for publication (300 DPI)
    print("\n[STEP 2] Enhancing for publication quality (300 DPI)...")
    final_300dpi_path = os.path.join(output_dir, f"{base_output_name}_300dpi.png")
    enhance_for_publication(cropped_path, final_300dpi_path, target_dpi=300, scale_factor=2.0)

    # Step 3: Create ultra-high quality version (600 DPI for print)
    print("\n[STEP 3] Creating ultra-high quality version (600 DPI)...")
    final_600dpi_path = os.path.join(output_dir, f"{base_output_name}_600dpi.png")
    enhance_for_publication(cropped_path, final_600dpi_path, target_dpi=600, scale_factor=3.0)

    # Step 4: Create TIFF version (lossless, industry standard)
    print("\n[STEP 4] Creating TIFF version (lossless)...")
    tiff_path = os.path.join(output_dir, f"{base_output_name}_300dpi.tiff")
    img = Image.open(final_300dpi_path)
    img.save(tiff_path, dpi=(300, 300), compression='lzw')

    print("\n" + "="*70)
    print("ENHANCEMENT COMPLETE!")
    print("="*70)
    print("\nOutput files created:")
    print(f"1. Cropped version: {cropped_path}")
    print(f"2. 300 DPI PNG (recommended for most journals): {final_300dpi_path}")
    print(f"3. 600 DPI PNG (ultra-high quality): {final_600dpi_path}")
    print(f"4. 300 DPI TIFF (lossless, industry standard): {tiff_path}")

    # File size comparison
    print("\nFile sizes:")
    for path, label in [
        (input_path, "Original"),
        (cropped_path, "Cropped"),
        (final_300dpi_path, "300 DPI PNG"),
        (final_600dpi_path, "600 DPI PNG"),
        (tiff_path, "300 DPI TIFF")
    ]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {label}: {size_mb:.2f} MB")

    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR Q1 JOURNAL SUBMISSION:")
    print("="*70)
    print("✓ Use '300dpi.png' for electronic submission (best quality/size ratio)")
    print("✓ Use '300dpi.tiff' if journal requires TIFF format")
    print("✓ Use '600dpi.png' for print-quality or if journal requires 600 DPI")
    print("✓ All versions have whitespace removed for professional appearance")
    print("✓ Colors and text preserved for readability")
    print("="*70)


if __name__ == "__main__":
    # Input file
    input_image = r"C:\Users\MyPC PRO\Documents\hello_world\luaran\figures\pipeline_architecture_horizontal.png"

    # Check if file exists
    if not os.path.exists(input_image):
        print(f"Error: File not found: {input_image}")
        exit(1)

    # Process the image
    create_publication_ready_figure(input_image)

    print("\n✅ All done! Check the 'figures' folder for enhanced versions.")
