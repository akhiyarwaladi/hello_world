#!/usr/bin/env python3
"""
IMPROVED Grad-CAM Visualization for Classification Models
Implements 4 key improvements based on paper analysis:
1. Multi-Layer Grad-CAM (higher resolution)
2. CLAHE Preprocessing (match paper)
3. Better Upsampling (bilateral filter)
4. Grad-CAM++ (better localization)
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# CLAHE PREPROCESSING (Paper Section 4.1)
# ============================================================================

def apply_clahe_preprocessing(image_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    as mentioned in paper Section 4.1, Page 11

    Args:
        image_pil: PIL Image
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        PIL Image with CLAHE applied
    """
    # Convert PIL to numpy
    img_array = np.array(image_pil)

    # Convert to LAB color space
    img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])

    # Convert back to RGB
    img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img_clahe)


# ============================================================================
# IMPROVED UPSAMPLING (Bilateral Filter)
# ============================================================================

def upsample_cam_high_quality(cam, target_size, method='bilateral'):
    """
    High-quality CAM upsampling with edge preservation

    Args:
        cam: Low-resolution CAM (H, W) in [0, 1]
        target_size: (width, height) tuple
        method: 'bilateral' or 'cubic'

    Returns:
        High-quality upsampled CAM
    """
    # Convert to uint8 for OpenCV
    cam_uint8 = (cam * 255).astype(np.uint8)

    # Bicubic interpolation (better than bilinear)
    cam_upsampled = cv2.resize(cam_uint8, target_size,
                               interpolation=cv2.INTER_CUBIC)

    if method == 'bilateral':
        # Apply bilateral filter to smooth while preserving edges
        cam_upsampled = cv2.bilateralFilter(cam_upsampled, d=9,
                                            sigmaColor=75, sigmaSpace=75)

    # Convert back to float [0,1]
    cam_final = cam_upsampled.astype(np.float32) / 255.0

    return cam_final


# ============================================================================
# STANDARD GRAD-CAM
# ============================================================================

class GradCAM:
    """Standard Grad-CAM implementation"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

        if class_idx is None:
            class_idx = predicted_class.item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), predicted_class.item(), confidence.item()


# ============================================================================
# GRAD-CAM++ (Better Localization)
# ============================================================================

class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation
    Better localization especially for multiple objects
    Reference: "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

        if class_idx is None:
            class_idx = predicted_class.item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM++ computation
        grads = self.gradients  # (1, C, H, W)
        acts = self.activations  # (1, C, H, W)

        # Compute alpha (Grad-CAM++ specific)
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads

        sum_acts = acts.sum(dim=(2, 3), keepdim=True)

        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + 1e-8

        alpha = alpha_num / alpha_denom

        # Compute weights
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * acts).sum(dim=1).squeeze()

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), predicted_class.item(), confidence.item()


# ============================================================================
# MULTI-LAYER SUPPORT
# ============================================================================

def get_multiple_target_layers(model, model_name):
    """
    Get layers at different depths for multi-resolution Grad-CAM
    Returns dict with 'early', 'mid', 'late' keys
    """
    model_name_lower = model_name.lower()

    if 'densenet' in model_name_lower:
        # DenseNet layers at different depths
        return {
            'early': model.features.denseblock2.denselayer12.conv2,   # ~28×28
            'mid': model.features.denseblock3.denselayer24.conv2,     # ~14×14
            'late': model.features.denseblock4.denselayer16.conv2     # ~7×7
        }
    elif 'efficientnet' in model_name_lower:
        # EfficientNet blocks
        n_blocks = len(model.features)
        return {
            'early': model.features[max(2, n_blocks//3)],      # Early block
            'mid': model.features[max(4, 2*n_blocks//3)],      # Mid block
            'late': model.features[-1]                          # Last block
        }
    elif 'resnet' in model_name_lower:
        # ResNet layers
        if hasattr(model.layer4[-1], 'conv3'):
            # ResNet50/101/152
            return {
                'early': model.layer2[-1].conv3,    # ~56×56
                'mid': model.layer3[-1].conv3,      # ~28×28
                'late': model.layer4[-1].conv3      # ~7×7
            }
        else:
            # ResNet18/34
            return {
                'early': model.layer2[-1].conv2,    # ~56×56
                'mid': model.layer3[-1].conv2,      # ~28×28
                'late': model.layer4[-1].conv2      # ~7×7
            }
    elif 'convnext' in model_name_lower:
        return {
            'early': model.features[2],
            'mid': model.features[4],
            'late': model.features[-1]
        }
    elif 'mobilenet' in model_name_lower:
        n_features = len(model.features)
        return {
            'early': model.features[max(4, n_features//3)],
            'mid': model.features[max(8, 2*n_features//3)],
            'late': model.features[-1]
        }
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_classification_model(model_path, num_classes, device):
    """Load trained classification model"""

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_name = checkpoint.get('model_name', 'densenet121')

    print(f"   Architecture: {model_name}")

    # Create model
    if 'efficientnet_b0' in model_name.lower():
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'efficientnet_b1' in model_name.lower():
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'efficientnet_b2' in model_name.lower():
        model = models.efficientnet_b2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'resnet50' in model_name.lower():
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'resnet101' in model_name.lower():
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'convnext' in model_name.lower():
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif 'mobilenet' in model_name.lower():
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:  # Default: DenseNet121
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, model_name


# ============================================================================
# VISUALIZATION
# ============================================================================

def apply_colormap_on_image(org_img, activation_map, colormap='jet', alpha=0.5):
    """Overlay heatmap on original image"""
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(activation_map)[:, :, :3]
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


def visualize_improved_gradcam(image_path, model, model_name, class_names, device,
                               output_dir, use_clahe=True, use_gradcam_pp=True):
    """
    Generate improved Grad-CAM visualization with all enhancements

    Args:
        image_path: Path to input image
        model: Trained model
        model_name: Model architecture name
        class_names: List of class names
        device: torch.device
        output_dir: Output directory
        use_clahe: Apply CLAHE preprocessing
        use_gradcam_pp: Use Grad-CAM++ instead of standard Grad-CAM
    """

    print(f"\n{'='*80}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*80}")

    # Load image
    img_pil = Image.open(image_path).convert('RGB')
    img_original = np.array(img_pil)

    # Apply CLAHE if requested
    if use_clahe:
        print("[1/4] Applying CLAHE preprocessing...")
        img_pil_processed = apply_clahe_preprocessing(img_pil, clip_limit=2.0)
        img_for_display = np.array(img_pil_processed)
    else:
        img_pil_processed = img_pil
        img_for_display = img_original

    # Transform for model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img_pil_processed).unsqueeze(0).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    conf_score = confidence.item()

    print(f"   Prediction: {predicted_class} ({conf_score:.2%})")

    # Get multi-layer targets
    print("[2/4] Getting multi-layer targets...")
    try:
        layer_dict = get_multiple_target_layers(model, model_name)
        layer_names = ['early', 'mid', 'late']
    except Exception as e:
        print(f"   Warning: Could not get multi-layer targets: {e}")
        print("   Falling back to single layer...")
        # Fallback to single layer
        if 'densenet' in model_name.lower():
            layer_dict = {'late': model.features.denseblock4.denselayer16.conv2}
        elif 'resnet' in model_name.lower():
            layer_dict = {'late': model.layer4[-1].conv3 if hasattr(model.layer4[-1], 'conv3') else model.layer4[-1].conv2}
        elif 'efficientnet' in model_name.lower():
            layer_dict = {'late': model.features[-1]}
        else:
            layer_dict = {'late': model.features[-1]}
        layer_names = ['late']

    print(f"   Layers: {layer_names}")

    # Generate Grad-CAM for each layer
    print(f"[3/4] Generating {'Grad-CAM++' if use_gradcam_pp else 'Grad-CAM'}...")
    cams = {}

    for layer_name in layer_names:
        target_layer = layer_dict[layer_name]

        if use_gradcam_pp:
            gradcam = GradCAMPlusPlus(model, target_layer)
        else:
            gradcam = GradCAM(model, target_layer)

        cam, _, _ = gradcam.generate(img_tensor, class_idx=predicted_idx.item())

        # High-quality upsampling
        cam_upsampled = upsample_cam_high_quality(
            cam,
            (img_for_display.shape[1], img_for_display.shape[0]),
            method='bilateral'
        )

        cams[layer_name] = cam_upsampled
        print(f"   Layer '{layer_name}': CAM shape {cam.shape} -> {cam_upsampled.shape}")

    # Create comprehensive visualization
    print("[4/4] Creating visualizations...")

    # Figure 1: Multi-layer comparison
    n_layers = len(layer_names)
    fig, axes = plt.subplots(2, n_layers + 1, figsize=(5 * (n_layers + 1), 10))

    fig.suptitle(
        f'Multi-Layer {"Grad-CAM++" if use_gradcam_pp else "Grad-CAM"} Visualization\n'
        f'Predicted: {predicted_class} (Confidence: {conf_score:.2%})',
        fontsize=16, fontweight='bold'
    )

    # Original image (first column)
    if n_layers == 1:
        axes[0].imshow(img_original)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(img_for_display)
        axes[1].set_title('With CLAHE' if use_clahe else 'Processed', fontsize=12, fontweight='bold')
        axes[1].axis('off')
    else:
        axes[0, 0].imshow(img_original)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[1, 0].imshow(img_for_display)
        axes[1, 0].set_title('With CLAHE' if use_clahe else 'Processed', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

    # Each layer's Grad-CAM
    for idx, layer_name in enumerate(layer_names):
        cam = cams[layer_name]
        overlay, heatmap = apply_colormap_on_image(img_for_display, cam, alpha=0.5)

        col_idx = idx + 1

        if n_layers == 1:
            # Heatmap
            axes[0].imshow(heatmap)
            axes[0].set_title(f'Heatmap ({layer_name.title()})', fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # Overlay
            axes[1].imshow(overlay)
            axes[1].set_title(f'Overlay ({layer_name.title()})', fontsize=12, fontweight='bold')
            axes[1].axis('off')
        else:
            # Heatmap
            axes[0, col_idx].imshow(heatmap)
            axes[0, col_idx].set_title(f'Heatmap ({layer_name.title()})', fontsize=12, fontweight='bold')
            axes[0, col_idx].axis('off')

            # Overlay
            axes[1, col_idx].imshow(overlay)
            axes[1, col_idx].set_title(f'Overlay ({layer_name.title()})', fontsize=12, fontweight='bold')
            axes[1, col_idx].axis('off')

    plt.tight_layout()

    # Save main figure
    output_file = Path(output_dir) / f'improved_gradcam_{Path(image_path).stem}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   [SAVED] Main visualization: {output_file}")

    # Save individual components
    components_dir = Path(output_dir) / 'components'
    components_dir.mkdir(parents=True, exist_ok=True)

    for layer_name in layer_names:
        cam = cams[layer_name]
        overlay, heatmap = apply_colormap_on_image(img_for_display, cam, alpha=0.5)

        # Save heatmap
        heatmap_file = components_dir / f'{Path(image_path).stem}_{layer_name}_heatmap.png'
        cv2.imwrite(str(heatmap_file), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

        # Save overlay
        overlay_file = components_dir / f'{Path(image_path).stem}_{layer_name}_overlay.png'
        cv2.imwrite(str(overlay_file), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"   [SAVED] Components: {components_dir}")

    return predicted_idx.item(), conf_score


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate IMPROVED Grad-CAM visualizations with multi-layer support'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained classification model (.pt)')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--class-names', type=str, nargs='+',
                       help='Class names (if not auto-detected)')
    parser.add_argument('--output', type=str, default='improved_gradcam_output',
                       help='Output directory')
    parser.add_argument('--max-images', type=int, default=20,
                       help='Maximum number of images to process')
    parser.add_argument('--no-clahe', action='store_true',
                       help='Disable CLAHE preprocessing')
    parser.add_argument('--use-standard-gradcam', action='store_true',
                       help='Use standard Grad-CAM instead of Grad-CAM++')

    args = parser.parse_args()

    print("="*80)
    print("IMPROVED GRAD-CAM VISUALIZATION")
    print("="*80)
    print("Improvements:")
    print("  [+] Multi-layer Grad-CAM (higher resolution)")
    print("  [+] CLAHE preprocessing (paper-matched)")
    print("  [+] Bilateral filtering upsampling")
    print(f"  [+] {'Grad-CAM++' if not args.use_standard_gradcam else 'Standard Grad-CAM'}")
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
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
        else:
            # Infer from model
            try:
                if 'densenet' in checkpoint.get('model_name', '').lower():
                    num_classes = checkpoint['model_state_dict']['classifier.weight'].shape[0]
                elif 'resnet' in checkpoint.get('model_name', '').lower():
                    num_classes = checkpoint['model_state_dict']['fc.weight'].shape[0]
                elif 'efficientnet' in checkpoint.get('model_name', '').lower():
                    num_classes = checkpoint['model_state_dict']['classifier.1.weight'].shape[0]
                else:
                    num_classes = 4
                class_names = [f'Class_{i}' for i in range(num_classes)]
            except:
                class_names = ['gametocyte', 'ring', 'schizont', 'trophozoite']

    print(f"Classes ({len(class_names)}): {class_names}")

    # Load model
    print("\n[LOAD] Loading model...")
    model, model_name = load_classification_model(args.model, len(class_names), device)

    # Get image paths
    images_path = Path(args.images)
    if images_path.is_file():
        image_files = [images_path]
    elif images_path.is_dir():
        # Use recursive glob to find images in subdirectories too
        image_files = sorted(
            list(images_path.glob("**/*.jpg")) +
            list(images_path.glob("**/*.JPG")) +
            list(images_path.glob("**/*.png")) +
            list(images_path.glob("**/*.PNG"))
        )[:args.max_images]
    else:
        print(f"[ERROR] Invalid path: {args.images}")
        return

    print(f"\n[FOUND] {len(image_files)} images")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process images
    print("\n[PROCESS] Generating improved Grad-CAM visualizations...")
    print("="*80)

    results = []
    for i, img_path in enumerate(image_files, 1):
        try:
            pred_idx, conf = visualize_improved_gradcam(
                img_path, model, model_name, class_names, device,
                output_path,
                use_clahe=not args.no_clahe,
                use_gradcam_pp=not args.use_standard_gradcam
            )
            results.append({
                'image': img_path.name,
                'predicted': class_names[pred_idx],
                'confidence': conf
            })
        except Exception as e:
            print(f"\n[ERROR] Failed to process {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("IMPROVED GRAD-CAM GENERATION COMPLETED!")
    print("="*80)
    print(f"[OK] Processed: {len(results)}/{len(image_files)} images")
    print(f"[OK] Output: {output_path}")
    print(f"[OK] Components: {output_path}/components/")
    print("\nSummary:")
    for r in results:
        print(f"  - {r['image']}: {r['predicted']} ({r['confidence']:.2%})")
    print("="*80)


if __name__ == "__main__":
    main()
