#!/usr/bin/env python3
"""
Advanced Model Interpretation and Visualization for Malaria Detection
Implements GradCAM, feature visualization, attention maps, and model comparison tools
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import argparse
from datetime import datetime
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class GradCAM:
    """GradCAM implementation for visualization"""

    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def generate_cam(self, input_image, class_idx=None):
        """Generate GradCAM heatmap"""
        self.model.eval()

        # Forward pass
        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()

        # Generate CAM
        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam / torch.max(cam)

        return cam.detach().numpy()

class ModelInterpreter:
    """Comprehensive model interpretation toolkit"""

    def __init__(self, model_path, data_path, output_dir="interpretation_results"):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load model
        self.model = self._load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Data setup
        self.test_loader = self._setup_data()

        print(f"🔍 Model Interpreter initialized")
        print(f"📂 Output directory: {self.output_dir}")
        print(f"💻 Device: {self.device}")

    def _load_model(self):
        """Load trained model"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')

            if 'model_state_dict' in checkpoint:
                # PyTorch checkpoint format
                model_name = checkpoint.get('model_name', 'resnet18')
                class_names = checkpoint.get('class_names', ['P_falciparum', 'P_vivax', 'P_malariae', 'P_ovale', 'Mixed_infection', 'Uninfected'])

                # Recreate model architecture
                model = self._create_model(model_name, len(class_names))
                model.load_state_dict(checkpoint['model_state_dict'])

                self.class_names = class_names

            else:
                # Direct model format
                model = checkpoint
                self.class_names = ['P_falciparum', 'P_vivax', 'P_malariae', 'P_ovale', 'Mixed_infection', 'Uninfected']

            model.eval()
            return model

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def _create_model(self, model_name, num_classes):
        """Recreate model architecture"""
        from torchvision import models
        import torch.nn as nn

        if model_name.startswith('resnet'):
            if '18' in model_name:
                model = models.resnet18()
            elif '50' in model_name:
                model = models.resnet50()
            else:
                model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name.startswith('efficientnet'):
            model = models.efficientnet_b0()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        elif model_name.startswith('densenet'):
            model = models.densenet121()
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)

        elif model_name.startswith('mobilenet'):
            model = models.mobilenet_v2()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        else:
            # Default to ResNet18
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    def _setup_data(self):
        """Setup test data loader"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        try:
            # Try to load as ImageFolder
            test_dataset = datasets.ImageFolder(
                root=self.data_path / "test",
                transform=transform
            )
        except:
            # Try to load from single directory
            test_dataset = datasets.ImageFolder(
                root=self.data_path,
                transform=transform
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

        return test_loader

    def generate_gradcam_analysis(self, num_samples=20):
        """Generate GradCAM analysis for sample images"""
        print("🔥 Generating GradCAM analysis...")

        # Determine target layer based on model architecture
        target_layer = self._get_target_layer()

        if not target_layer:
            print("❌ Could not determine target layer for GradCAM")
            return

        gradcam = GradCAM(self.model, target_layer)

        gradcam_dir = self.output_dir / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)

        sample_count = 0
        class_samples = {name: 0 for name in self.class_names}
        max_per_class = max(1, num_samples // len(self.class_names))

        for batch_idx, (data, target) in enumerate(self.test_loader):
            if sample_count >= num_samples:
                break

            class_name = self.class_names[target.item()]
            if class_samples[class_name] >= max_per_class:
                continue

            data = data.to(self.device)
            target = target.to(self.device)

            # Generate CAM
            cam = gradcam.generate_cam(data, target.item())

            # Create visualization
            self._create_gradcam_visualization(
                data.cpu().squeeze(),
                cam,
                class_name,
                gradcam_dir / f"gradcam_{class_name}_{class_samples[class_name]:03d}.png"
            )

            class_samples[class_name] += 1
            sample_count += 1

        print(f"✅ Generated {sample_count} GradCAM visualizations")

    def _get_target_layer(self):
        """Determine appropriate target layer for GradCAM"""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = name
        return target_layer

    def _create_gradcam_visualization(self, image, cam, class_name, save_path):
        """Create and save GradCAM visualization"""
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)

        # Convert to numpy
        image_np = image.permute(1, 2, 0).numpy()

        # Resize CAM to match image
        cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(cam_resized, cmap='hot')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(image_np)
        axes[2].imshow(cam_resized, cmap='hot', alpha=0.4)
        axes[2].set_title(f'Overlay - {class_name}')
        axes[2].axis('off')

        plt.suptitle(f'GradCAM Analysis: {class_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def analyze_feature_representations(self, num_samples=100):
        """Analyze learned feature representations using t-SNE"""
        print("🧠 Analyzing feature representations...")

        features = []
        labels = []

        # Hook to extract features
        feature_extractor = {}
        def get_features(name):
            def hook(model, input, output):
                feature_extractor[name] = output.detach()
            return hook

        # Register hook at the last layer before classification
        last_layer_name = self._get_feature_layer()
        if last_layer_name:
            for name, module in self.model.named_modules():
                if name == last_layer_name:
                    module.register_forward_hook(get_features(name))
                    break

        # Extract features
        sample_count = 0
        for batch_idx, (data, target) in enumerate(self.test_loader):
            if sample_count >= num_samples:
                break

            data = data.to(self.device)

            with torch.no_grad():
                _ = self.model(data)

                if last_layer_name in feature_extractor:
                    feat = feature_extractor[last_layer_name]
                    if len(feat.shape) > 2:
                        feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze()
                    features.append(feat.cpu().numpy().flatten())
                    labels.append(target.item())

            sample_count += 1

        if not features:
            print("❌ Could not extract features")
            return

        features = np.array(features)
        labels = np.array(labels)

        # Apply t-SNE
        print("📊 Applying t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        features_2d = tsne.fit_transform(features)

        # Create visualization
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))

        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            if np.any(mask):
                plt.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=[colors[i]],
                    label=class_name,
                    alpha=0.7,
                    s=50
                )

        plt.title('t-SNE Visualization of Learned Features', fontsize=16)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_tsne.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Also create PCA visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)

        plt.figure(figsize=(12, 8))
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            if np.any(mask):
                plt.scatter(
                    features_pca[mask, 0],
                    features_pca[mask, 1],
                    c=[colors[i]],
                    label=class_name,
                    alpha=0.7,
                    s=50
                )

        plt.title('PCA Visualization of Learned Features', fontsize=16)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_pca.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Feature analysis complete with {len(features)} samples")

    def _get_feature_layer(self):
        """Get the layer name for feature extraction"""
        layers = list(self.model.named_modules())

        # Find the last convolutional or linear layer before the classifier
        for name, module in reversed(layers):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.AdaptiveAvgPool2d)):
                return name

        return None

    def generate_prediction_confidence_analysis(self):
        """Analyze model prediction confidence across classes"""
        print("📊 Analyzing prediction confidence...")

        all_predictions = []
        all_confidences = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)

                probabilities = F.softmax(output, dim=1)
                max_prob, predicted = torch.max(probabilities, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_confidences.extend(max_prob.cpu().numpy())
                all_labels.extend(target.numpy())

        # Create confidence analysis
        df = pd.DataFrame({
            'true_label': all_labels,
            'predicted_label': all_predictions,
            'confidence': all_confidences,
            'correct': np.array(all_labels) == np.array(all_predictions)
        })

        # Map labels to class names
        df['true_class'] = [self.class_names[i] for i in df['true_label']]
        df['predicted_class'] = [self.class_names[i] for i in df['predicted_label']]

        # Confidence distribution by class
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        for i, class_name in enumerate(self.class_names):
            class_confidences = df[df['true_label'] == i]['confidence']
            if len(class_confidences) > 0:
                plt.hist(class_confidences, bins=20, alpha=0.7, label=class_name)
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by True Class')
        plt.legend()

        plt.subplot(2, 2, 2)
        correct_conf = df[df['correct'] == True]['confidence']
        incorrect_conf = df[df['correct'] == False]['confidence']
        plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution: Correct vs Incorrect')
        plt.legend()

        # Confidence vs accuracy by class
        plt.subplot(2, 2, 3)
        class_stats = []
        for i, class_name in enumerate(self.class_names):
            class_df = df[df['true_label'] == i]
            if len(class_df) > 0:
                accuracy = class_df['correct'].mean()
                mean_confidence = class_df['confidence'].mean()
                class_stats.append({'class': class_name, 'accuracy': accuracy, 'confidence': mean_confidence})

        if class_stats:
            stats_df = pd.DataFrame(class_stats)
            plt.scatter(stats_df['confidence'], stats_df['accuracy'])
            for idx, row in stats_df.iterrows():
                plt.annotate(row['class'], (row['confidence'], row['accuracy']))
            plt.xlabel('Mean Confidence')
            plt.ylabel('Accuracy')
            plt.title('Confidence vs Accuracy by Class')

        # Calibration plot
        plt.subplot(2, 2, 4)
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_confidences = []

        for i in range(len(bins)-1):
            mask = (df['confidence'] >= bins[i]) & (df['confidence'] < bins[i+1])
            if mask.sum() > 0:
                bin_acc = df[mask]['correct'].mean()
                bin_conf = df[mask]['confidence'].mean()
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)

        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        if bin_confidences and bin_accuracies:
            plt.plot(bin_confidences, bin_accuracies, 'ro-', label='Model')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Calibration Plot')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Save detailed results
        df.to_csv(self.output_dir / "prediction_analysis.csv", index=False)

        print(f"✅ Confidence analysis complete with {len(df)} predictions")

    def generate_comprehensive_report(self):
        """Generate comprehensive interpretation report"""
        print("📋 Generating comprehensive report...")

        report_content = f"""# Model Interpretation Report

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model:** {self.model_path}
**Data:** {self.data_path}

## Model Architecture

The analyzed model contains the following structure:
- **Classes:** {len(self.class_names)} ({', '.join(self.class_names)})
- **Device:** {self.device}

## Analysis Components

### 1. GradCAM Visualization
- **Purpose:** Understand which regions of the image the model focuses on
- **Output:** Individual heatmaps for each class showing attention regions
- **Files:** `gradcam/gradcam_*.png`

### 2. Feature Representation Analysis
- **Purpose:** Visualize how the model clusters different classes in feature space
- **Methods:** t-SNE and PCA dimensionality reduction
- **Files:** `feature_tsne.png`, `feature_pca.png`

### 3. Prediction Confidence Analysis
- **Purpose:** Understand model certainty and calibration
- **Metrics:** Confidence distributions, calibration curves
- **Files:** `confidence_analysis.png`, `prediction_analysis.csv`

## Key Insights

1. **Attention Patterns:** GradCAM reveals which microscopic features the model considers most important for classification
2. **Feature Clustering:** t-SNE/PCA show how well the model separates different malaria species
3. **Model Calibration:** Confidence analysis indicates how well the model's certainty matches its accuracy

## Usage Recommendations

- Review GradCAM heatmaps to ensure the model focuses on relevant cellular features
- Check feature clustering to identify potential confusion between similar species
- Use confidence analysis to set appropriate decision thresholds for deployment

---

*Generated by Model Interpretation Toolkit for Malaria Detection Research*
"""

        with open(self.output_dir / "interpretation_report.md", 'w') as f:
            f.write(report_content)

        print(f"📋 Report saved to {self.output_dir}/interpretation_report.md")

def main():
    parser = argparse.ArgumentParser(description="Model Interpretation for Malaria Detection")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--data", required=True, help="Path to test data")
    parser.add_argument("--output", default="interpretation_results", help="Output directory")
    parser.add_argument("--gradcam_samples", type=int, default=20, help="Number of GradCAM samples")
    parser.add_argument("--feature_samples", type=int, default=100, help="Number of samples for feature analysis")

    args = parser.parse_args()

    print("=" * 60)
    print("MODEL INTERPRETATION FOR MALARIA DETECTION")
    print("=" * 60)

    # Initialize interpreter
    interpreter = ModelInterpreter(args.model, args.data, args.output)

    if interpreter.model is None:
        print("❌ Failed to load model")
        return

    try:
        # Run all analyses
        interpreter.generate_gradcam_analysis(args.gradcam_samples)
        interpreter.analyze_feature_representations(args.feature_samples)
        interpreter.generate_prediction_confidence_analysis()

        # Generate report
        interpreter.generate_comprehensive_report()

        print("\n🎉 Model interpretation complete!")
        print(f"📂 Results available in: {args.output}")

    except Exception as e:
        print(f"❌ Error during interpretation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()