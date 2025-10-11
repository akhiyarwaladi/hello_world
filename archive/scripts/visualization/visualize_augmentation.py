#!/usr/bin/env python3
"""
Visualize Augmentation Techniques for Paper
Generate examples of augmented parasite images from ground truth crops
"""

import os
import sys
import random
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class RotationWithEdgePadding:
    """Custom rotation transform with edge replication instead of black padding"""
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        # Pad image with edge replication before rotation
        # Calculate padding needed (for 15° rotation, ~10% padding is enough)
        pad_size = int(img.size[0] * 0.15)  # 15% padding for safety

        # Pad with edge replication
        padded = TF.pad(img, padding=pad_size, padding_mode='edge')

        # Rotate the padded image
        rotated = TF.rotate(padded, self.degrees, interpolation=InterpolationMode.BILINEAR)

        # Center crop back to original size
        rotated_cropped = TF.center_crop(rotated, img.size)

        return rotated_cropped

def get_individual_augmentations(image_size=224):
    """Get individual augmentation transforms for visualization"""

    # Base transform (no augmentation)
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    # Individual augmentations
    augmentations = {
        '1_original': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]),

        '2_horizontal_flip': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),  # Force flip
            transforms.ToTensor()
        ]),

        '3_vertical_flip': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomVerticalFlip(p=1.0),  # Force flip
            transforms.ToTensor()
        ]),

        '4_rotation_15deg': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RotationWithEdgePadding(15),  # Custom rotation with edge padding
            transforms.ToTensor()
        ]),

        '5_rotation_minus15deg': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RotationWithEdgePadding(-15),  # Custom rotation with edge padding
            transforms.ToTensor()
        ]),

        '6_brightness_increase': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0),
            transforms.ToTensor()
        ]),

        '7_brightness_decrease': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=(0.5, 0.5), contrast=0, saturation=0, hue=0),  # (0.5, 0.5) = multiply by 0.5 = darker
            transforms.ToTensor()
        ]),

        '8_contrast_increase': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0, contrast=0.3, saturation=0, hue=0),
            transforms.ToTensor()
        ]),

        '9_saturation_increase': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0.3, hue=0),
            transforms.ToTensor()
        ]),

        '10_combined_augment': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            RotationWithEdgePadding(10),  # Custom rotation with edge padding
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor()
        ])
    }

    return augmentations

def tensor_to_image(tensor):
    """Convert tensor to numpy image for visualization"""
    # Convert from CxHxW to HxWxC
    img = tensor.permute(1, 2, 0).cpu().numpy()
    # Clip values to [0, 1]
    img = np.clip(img, 0, 1)
    return img

def visualize_augmentations(image_path, output_dir, image_size=224):
    """Visualize all augmentations for a single parasite image"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load image
    image = Image.open(image_path).convert('RGB')
    parasite_name = Path(image_path).stem
    parasite_class = Path(image_path).parent.name

    print(f"\n[VISUALIZING] {parasite_class}/{parasite_name}")

    # Get augmentations
    augmentations = get_individual_augmentations(image_size)

    # Apply each augmentation
    augmented_images = {}
    for aug_name, transform in augmentations.items():
        # Set random seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)

        # Apply transform
        img_tensor = transform(image)
        augmented_images[aug_name] = tensor_to_image(img_tensor)

    # Create visualization grid
    n_augmentations = len(augmented_images)
    n_cols = 5
    n_rows = (n_augmentations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    fig.suptitle(f'Augmentation Examples: {parasite_class} - {parasite_name}',
                 fontsize=16, fontweight='bold', y=0.995)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    # Plot each augmentation
    for idx, (aug_name, img) in enumerate(augmented_images.items()):
        ax = axes_flat[idx]
        ax.imshow(img)

        # Format title (remove prefix number and underscore)
        title = aug_name.split('_', 1)[1].replace('_', ' ').title()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    # Hide empty subplots
    for idx in range(len(augmented_images), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()

    # Save figure
    output_file = output_path / f'augmentation_examples_{parasite_class}_{parasite_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {output_file}")

    # Save individual augmented images
    individual_dir = output_path / 'individual_augmentations' / parasite_class / parasite_name
    individual_dir.mkdir(parents=True, exist_ok=True)

    for aug_name, img in augmented_images.items():
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil.save(individual_dir / f'{aug_name}.png')

    print(f"[SAVED] Individual images to {individual_dir}")

    return output_file

def create_comparison_figure(image_path, output_dir, image_size=224):
    """Create a compact comparison figure showing original + key augmentations"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load image
    image = Image.open(image_path).convert('RGB')
    parasite_name = Path(image_path).stem
    parasite_class = Path(image_path).parent.name

    # Key augmentations for paper
    key_augmentations = {
        'Original': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]),
        'Horizontal Flip': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ]),
        'Rotation +15°': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RotationWithEdgePadding(15),  # Custom rotation with edge padding
            transforms.ToTensor()
        ]),
        'Rotation -15°': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RotationWithEdgePadding(-15),  # Custom rotation with edge padding
            transforms.ToTensor()
        ]),
        'Brightness↑': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.3),
            transforms.ToTensor()
        ]),
        'Combined': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            RotationWithEdgePadding(10),  # Custom rotation with edge padding
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
    }

    # Apply augmentations
    augmented_images = {}
    for aug_name, transform in key_augmentations.items():
        random.seed(42)
        torch.manual_seed(42)
        img_tensor = transform(image)
        augmented_images[aug_name] = tensor_to_image(img_tensor)

    # Create compact figure (2 rows x 3 cols)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Data Augmentation Pipeline\n{parasite_class}',
                 fontsize=14, fontweight='bold')

    axes_flat = axes.flatten()

    for idx, (aug_name, img) in enumerate(augmented_images.items()):
        ax = axes_flat[idx]
        ax.imshow(img)
        ax.set_title(aug_name, fontsize=11, fontweight='bold')
        ax.axis('off')

        # Add border for original
        if aug_name == 'Original':
            rect = mpatches.Rectangle((0, 0), img.shape[1]-1, img.shape[0]-1,
                                     linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    plt.tight_layout()

    # Save
    output_file = output_path / f'augmentation_comparison_{parasite_class}_{parasite_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] Comparison figure: {output_file}")

    return output_file

def main():
    parser = argparse.ArgumentParser(description='Visualize augmentation for paper')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to parasite crop image')
    parser.add_argument('--output', type=str, default='augmentation_examples',
                       help='Output directory')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size (default: 224)')
    parser.add_argument('--comparison-only', action='store_true',
                       help='Only create comparison figure (compact for paper)')

    args = parser.parse_args()

    print("="*80)
    print("AUGMENTATION VISUALIZATION FOR PAPER")
    print("="*80)
    print(f"Input image: {args.image}")
    print(f"Output directory: {args.output}")
    print(f"Image size: {args.image_size}x{args.image_size}")

    # Check if image exists
    if not Path(args.image).exists():
        print(f"[ERROR] Image not found: {args.image}")
        return

    # Create visualizations
    if args.comparison_only:
        print("\n[MODE] Creating comparison figure only (for paper)")
        create_comparison_figure(args.image, args.output, args.image_size)
    else:
        print("\n[MODE] Creating full augmentation examples")
        visualize_augmentations(args.image, args.output, args.image_size)
        print("\n[MODE] Creating comparison figure")
        create_comparison_figure(args.image, args.output, args.image_size)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETED!")
    print("="*80)
    print(f"Check output directory: {args.output}")

if __name__ == "__main__":
    main()
