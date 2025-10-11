#!/usr/bin/env python3
"""
Generate Grad-CAM Visualizations for Classification Models

Grad-CAM (Gradient-weighted Class Activation Mapping) shows which regions
of the image the model focuses on when making predictions.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class GradCAM:
    """Grad-CAM implementation"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index (if None, use predicted class)

        Returns:
            cam: Grad-CAM heatmap (H, W)
            predicted_class: Predicted class index
            confidence: Prediction confidence
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Get predicted class if not specified
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

        if class_idx is None:
            class_idx = predicted_class.item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), predicted_class.item(), confidence.item()


def get_target_layer(model, model_name):
    """Get the last convolutional layer for different architectures"""

    model_name_lower = model_name.lower()

    if 'densenet' in model_name_lower:
        # DenseNet: features.denseblock4.denselayer16.conv2
        return model.features.denseblock4.denselayer16.conv2
    elif 'efficientnet' in model_name_lower:
        # EfficientNet: features[-1] (last conv layer)
        return model.features[-1]
    elif 'resnet' in model_name_lower:
        # ResNet: layer4[-1].conv2 or layer4[-1].conv3
        if hasattr(model.layer4[-1], 'conv3'):
            return model.layer4[-1].conv3  # ResNet50/101/152
        else:
            return model.layer4[-1].conv2  # ResNet18/34
    elif 'convnext' in model_name_lower:
        # ConvNeXt: features[-1]
        return model.features[-1]
    elif 'mobilenet' in model_name_lower:
        # MobileNet: features[-1]
        return model.features[-1]
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def load_classification_model(model_path, num_classes, device):
    """Load trained classification model"""

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_name = checkpoint.get('model_name', 'densenet121')

    print(f"   Loading architecture: {model_name}")

    # Create model architecture
    if 'efficientnet_b0' in model_name.lower():
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    elif 'efficientnet_b1' in model_name.lower():
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    elif 'efficientnet_b2' in model_name.lower():
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    elif 'resnet50' in model_name.lower():
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif 'resnet101' in model_name.lower():
        model = models.resnet101(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif 'convnext' in model_name.lower():
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = torch.nn.Linear(
            model.classifier[2].in_features, num_classes
        )
    elif 'mobilenet' in model_name.lower():
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, num_classes
        )
    else:  # Default to DenseNet121
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(
            model.classifier.in_features, num_classes
        )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, model_name


def apply_colormap_on_image(org_img, activation_map, colormap='jet', alpha=0.5):
    """
    Overlay heatmap on original image

    Args:
        org_img: Original image (H, W, 3) in RGB
        activation_map: Heatmap (H, W) in [0, 1]
        colormap: Matplotlib colormap name
        alpha: Transparency of heatmap

    Returns:
        overlayed: Image with heatmap overlay
    """
    # Get colormap
    cmap = cm.get_cmap(colormap)

    # Apply colormap to heatmap
    heatmap = cmap(activation_map)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)

    # Resize heatmap to match image size
    if heatmap.shape[:2] != org_img.shape[:2]:
        heatmap = cv2.resize(heatmap, (org_img.shape[1], org_img.shape[0]))

    # Ensure org_img is uint8
    if org_img.dtype != np.uint8:
        org_img = (org_img * 255).astype(np.uint8)

    # Overlay
    overlayed = cv2.addWeighted(org_img, 1-alpha, heatmap, alpha, 0)

    return overlayed, heatmap


def visualize_gradcam(image_path, model, model_name, class_names, device,
                      output_dir, transform, show_multiple=False):
    """
    Generate Grad-CAM visualization for a single image

    Args:
        image_path: Path to input image
        model: Trained classification model
        model_name: Model architecture name
        class_names: List of class names
        device: torch.device
        output_dir: Output directory
        transform: Image preprocessing transform
        show_multiple: If True, show Grad-CAM for all classes
    """

    # Load and preprocess image
    img_pil = Image.open(image_path).convert('RGB')
    img_array = np.array(img_pil)

    # Transform for model
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Get target layer
    target_layer = get_target_layer(model, model_name)

    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)

    # Generate Grad-CAM for predicted class
    cam, predicted_idx, confidence = gradcam.generate(img_tensor)

    # Resize CAM to match input image size
    cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))

    # Create overlay
    overlayed, heatmap = apply_colormap_on_image(img_array, cam_resized, alpha=0.5)

    # Create figure
    if show_multiple:
        # Show Grad-CAM for all classes
        n_classes = len(class_names)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        fig.suptitle(f'Grad-CAM for All Classes\n{Path(image_path).name}',
                     fontsize=14, fontweight='bold')

        axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for class_idx in range(n_classes):
            cam_class, _, _ = gradcam.generate(img_tensor, class_idx=class_idx)
            cam_class_resized = cv2.resize(cam_class, (img_array.shape[1], img_array.shape[0]))
            overlayed_class, _ = apply_colormap_on_image(img_array, cam_class_resized, alpha=0.5)

            ax = axes_flat[class_idx]
            ax.imshow(overlayed_class)

            title = f'{class_names[class_idx]}'
            if class_idx == predicted_idx:
                title += f'\n(Predicted: {confidence:.2f})'
                ax.set_title(title, fontsize=12, fontweight='bold', color='green')
            else:
                ax.set_title(title, fontsize=12)
            ax.axis('off')

        # Hide empty subplots
        for idx in range(n_classes, len(axes_flat)):
            axes_flat[idx].axis('off')

        plt.tight_layout()

        # Save
        output_file = Path(output_dir) / f'gradcam_all_classes_{Path(image_path).stem}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   [SAVED] Multi-class Grad-CAM: {output_file}")

    else:
        # Show only predicted class
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Grad-CAM Visualization\nPredicted: {class_names[predicted_idx]} (conf: {confidence:.2f})',
                     fontsize=14, fontweight='bold')

        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Heatmap only
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(overlayed)
        axes[2].set_title('Overlay (50% opacity)', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # Overlay with higher opacity
        overlayed_high, _ = apply_colormap_on_image(img_array, cam_resized, alpha=0.7)
        axes[3].imshow(overlayed_high)
        axes[3].set_title('Overlay (70% opacity)', fontsize=12, fontweight='bold')
        axes[3].axis('off')

        plt.tight_layout()

        # Save
        output_file = Path(output_dir) / f'gradcam_{class_names[predicted_idx]}_{Path(image_path).stem}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   [SAVED] Grad-CAM: {output_file}")

        # Also save individual components
        individual_dir = Path(output_dir) / 'gradcam_components'
        individual_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(individual_dir / f'{Path(image_path).stem}_heatmap.png'),
                    cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(individual_dir / f'{Path(image_path).stem}_overlay.png'),
                    cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))

    return predicted_idx, confidence


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained classification model (.pt)')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--class-names', type=str, nargs='+',
                       help='Class names (if not auto-detected)')
    parser.add_argument('--output', type=str, default='gradcam_output',
                       help='Output directory')
    parser.add_argument('--show-all-classes', action='store_true',
                       help='Show Grad-CAM for all classes')
    parser.add_argument('--max-images', type=int, default=10,
                       help='Maximum number of images to process')

    args = parser.parse_args()

    print("="*80)
    print("GRAD-CAM VISUALIZATION GENERATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Images: {args.images}")
    print(f"Output: {args.output}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Get class names
    if args.class_names:
        class_names = args.class_names
    else:
        # Try to infer from model checkpoint
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
        else:
            # Default class names
            num_classes = checkpoint['model_state_dict']['classifier.weight'].shape[0] if 'densenet' in checkpoint.get('model_name', '') else 4
            class_names = [f'Class_{i}' for i in range(num_classes)]

    print(f"Classes ({len(class_names)}): {class_names}")

    # Load model
    print("\n[LOAD] Loading model...")
    model, model_name = load_classification_model(args.model, len(class_names), device)
    print(f"   Model: {model_name}")

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Get image paths
    images_path = Path(args.images)
    if images_path.is_file():
        image_files = [images_path]
    elif images_path.is_dir():
        image_files = sorted(
            list(images_path.glob("*.jpg")) +
            list(images_path.glob("*.JPG")) +
            list(images_path.glob("*.png")) +
            list(images_path.glob("*.PNG"))
        )[:args.max_images]
    else:
        print(f"[ERROR] Invalid path: {args.images}")
        return

    print(f"\n[FOUND] {len(image_files)} images")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process images
    print("\n[PROCESS] Generating Grad-CAM visualizations...")
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {img_path.name}")
        try:
            pred_idx, conf = visualize_gradcam(
                img_path, model, model_name, class_names, device,
                output_path, transform, args.show_all_classes
            )
            print(f"   Predicted: {class_names[pred_idx]} ({conf:.2%})")
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")

    print("\n" + "="*80)
    print("GRAD-CAM GENERATION COMPLETED!")
    print("="*80)
    print(f"Output directory: {output_path}")
    print(f"Generated files:")
    print(f"  - Main visualizations: {len(image_files)} images")
    print(f"  - Component images: {output_path}/gradcam_components/")
    print("="*80)


if __name__ == "__main__":
    main()
