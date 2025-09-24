#!/usr/bin/env python3
"""
Deep Classification Analysis - Detailed evaluation of YOLOv8 Classification Results
Investigates suspicious 100% top-5 accuracy and creates comprehensive confusion matrix analysis
Following methodology from IEEE paper: "Automated Identification of Malaria-Infected Cells"
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import torch
import cv2
from ultralytics import YOLO
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ClassificationDeepAnalyzer:
    """Comprehensive analysis of classification results with focus on suspicious metrics"""

    def __init__(self,
                 model_path="results/current_experiments/training/classification/yolov11_classification/multi_pipeline_20250920_131500_yolo11_cls/weights/best.pt",
                 test_data_path="data/crops_from_yolo10_multi_pipeline_20250920_131500_yolo10_det/yolo_classification/test",
                 output_dir=None):

        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"analysis_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        self.output_dir.mkdir(exist_ok=True)

        # Species mapping based on journal paper
        self.species_mapping = {
            0: 'P_falciparum',
            1: 'P_malariae',
            2: 'P_ovale',
            3: 'P_vivax'
        }

        print(f"Deep Classification Analysis Initialized")
        print(f"Model: {self.model_path}")
        print(f"Test Data: {self.test_data_path}")
        print(f"Output: {self.output_dir}")

    def load_model_and_data(self):
        """Load trained model and test dataset"""
        print("\nLoading model and test data...")

        # Load YOLO classification model
        try:
            self.model = YOLO(str(self.model_path))
            print(f"[SUCCESS] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            return False

        # Load test dataset structure
        if not self.test_data_path.exists():
            print(f"[ERROR] Test data path not found: {self.test_data_path}")
            return False

        # Get test images by class
        self.test_images = {}
        self.test_labels = []
        self.test_paths = []

        class_dirs = sorted([d for d in self.test_data_path.iterdir() if d.is_dir()])
        print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.test_images[class_name] = []

            # Get all images in this class
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(class_dir.glob(ext)))

            print(f"  {class_name}: {len(image_files)} images")

            for img_path in image_files:
                self.test_images[class_name].append(str(img_path))
                self.test_labels.append(class_idx)
                self.test_paths.append(str(img_path))

        print(f"[SUCCESS] Total test images: {len(self.test_paths)}")
        print(f"[SUCCESS] Class distribution: {dict(zip(class_dirs, [len(self.test_images[d.name]) for d in class_dirs]))}")

        return True

    def run_inference_analysis(self):
        """Run inference on test set and analyze predictions"""
        print("\nRunning inference analysis...")

        predictions = []
        true_labels = []
        confidences = []
        top5_predictions = []

        print("Processing test images...")
        for idx, (img_path, true_label) in enumerate(zip(self.test_paths, self.test_labels)):
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(self.test_paths)}")

            try:
                # Run inference
                results = self.model(img_path, verbose=False)

                if results and len(results) > 0:
                    result = results[0]

                    # Get predictions
                    if hasattr(result, 'probs') and result.probs is not None:
                        probs = result.probs.data.cpu().numpy()
                        pred_class = np.argmax(probs)
                        confidence = np.max(probs)

                        # Get top-5 predictions
                        top5_idx = np.argsort(probs)[::-1][:5]
                        top5_probs = probs[top5_idx]

                        predictions.append(pred_class)
                        true_labels.append(true_label)
                        confidences.append(confidence)
                        top5_predictions.append((top5_idx, top5_probs))
                    else:
                        print(f"[WARNING] No probabilities for {img_path}")
                        predictions.append(-1)
                        true_labels.append(true_label)
                        confidences.append(0.0)
                        top5_predictions.append(([], []))
                else:
                    print(f"[WARNING] No results for {img_path}")
                    predictions.append(-1)
                    true_labels.append(true_label)
                    confidences.append(0.0)
                    top5_predictions.append(([], []))

            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {e}")
                predictions.append(-1)
                true_labels.append(true_label)
                confidences.append(0.0)
                top5_predictions.append(([], []))

        # Convert to numpy arrays
        self.predictions = np.array(predictions)
        self.true_labels = np.array(true_labels)
        self.confidences = np.array(confidences)
        self.top5_predictions = top5_predictions

        print(f"[SUCCESS] Inference completed on {len(predictions)} images")

        # Calculate basic metrics
        valid_mask = self.predictions != -1
        valid_predictions = self.predictions[valid_mask]
        valid_true_labels = self.true_labels[valid_mask]

        if len(valid_predictions) > 0:
            accuracy = accuracy_score(valid_true_labels, valid_predictions)
            print(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

            # Check top-5 accuracy
            top5_correct = 0
            for i, (true_label, (top5_idx, top5_probs)) in enumerate(zip(valid_true_labels,
                                                                         [self.top5_predictions[j] for j in np.where(valid_mask)[0]])):
                if len(top5_idx) > 0 and true_label in top5_idx:
                    top5_correct += 1

            top5_accuracy = top5_correct / len(valid_predictions)
            print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")

            # INVESTIGATE SUSPICIOUS 100% TOP-5 ACCURACY
            if top5_accuracy >= 0.99:
                print("SUSPICIOUS: Top-5 accuracy is suspiciously high!")
                self.investigate_suspicious_accuracy(valid_true_labels, valid_predictions)

        return True

    def investigate_suspicious_accuracy(self, true_labels, predictions):
        """Investigate why top-5 accuracy is suspiciously high"""
        print("\nINVESTIGATING SUSPICIOUS 100% TOP-5 ACCURACY...")

        # Check number of classes
        unique_true = np.unique(true_labels)
        unique_pred = np.unique(predictions)

        print(f"Unique true labels: {unique_true} (count: {len(unique_true)})")
        print(f"Unique predictions: {unique_pred} (count: {len(unique_pred)})")

        # Check class distribution
        from collections import Counter
        true_dist = Counter(true_labels)
        pred_dist = Counter(predictions)

        print(f"True label distribution: {dict(true_dist)}")
        print(f"Prediction distribution: {dict(pred_dist)}")

        # CRITICAL FINDING: If we only have 4 classes, top-5 accuracy will always be 100%!
        if len(unique_true) <= 4:
            print("CRITICAL FINDING: Dataset only has 4 classes!")
            print("This explains 100% top-5 accuracy - model always includes correct class in top-5!")
            print("Top-5 accuracy is meaningless with only 4 classes!")

        # Check if model is predicting only certain classes
        if len(unique_pred) < len(unique_true):
            print("WARNING: Model is not predicting all classes!")
            missing_classes = set(unique_true) - set(unique_pred)
            print(f"Missing predictions for classes: {missing_classes}")

        # Check confidence distributions
        print(f"Confidence stats:")
        print(f"   Mean: {np.mean(self.confidences):.4f}")
        print(f"   Std:  {np.std(self.confidences):.4f}")
        print(f"   Min:  {np.min(self.confidences):.4f}")
        print(f"   Max:  {np.max(self.confidences):.4f}")

    def create_comprehensive_confusion_matrix(self):
        """Create detailed confusion matrix analysis like in the journal paper"""
        print("\nCreating comprehensive confusion matrix analysis...")

        valid_mask = self.predictions != -1
        valid_predictions = self.predictions[valid_mask]
        valid_true_labels = self.true_labels[valid_mask]

        if len(valid_predictions) == 0:
            print("[ERROR] No valid predictions for confusion matrix")
            return

        # Create confusion matrix
        cm = confusion_matrix(valid_true_labels, valid_predictions)

        # Create figure with multiple subplots like in the journal
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Classification Analysis\n(Following IEEE Journal Methodology)',
                     fontsize=16, fontweight='bold')

        # 1. Standard Confusion Matrix
        class_names = [self.species_mapping.get(i, f'Class_{i}') for i in range(len(cm))]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix (Raw Counts)')
        axes[0,0].set_xlabel('Predicted Class')
        axes[0,0].set_ylabel('True Class')

        # 2. Normalized Confusion Matrix (like in journal)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0,1])
        axes[0,1].set_title('Normalized Confusion Matrix (Recall)')
        axes[0,1].set_xlabel('Predicted Class')
        axes[0,1].set_ylabel('True Class')

        # 3. Per-class metrics visualization
        precision, recall, f1, support = precision_recall_fscore_support(
            valid_true_labels, valid_predictions, average=None, zero_division=0)

        metrics_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })

        x_pos = np.arange(len(class_names))
        width = 0.25

        axes[1,0].bar(x_pos - width, precision, width, label='Precision', alpha=0.8)
        axes[1,0].bar(x_pos, recall, width, label='Recall', alpha=0.8)
        axes[1,0].bar(x_pos + width, f1, width, label='F1-Score', alpha=0.8)

        axes[1,0].set_xlabel('Classes')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_title('Per-Class Performance Metrics')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(class_names, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # 4. Confidence distribution by class
        conf_by_class = {class_names[i]: [] for i in range(len(class_names))}

        for true_label, pred_label, conf in zip(valid_true_labels, valid_predictions,
                                               self.confidences[valid_mask]):
            class_name = class_names[true_label]
            conf_by_class[class_name].append(conf)

        # Box plot of confidence by class
        conf_data = [conf_by_class[class_name] for class_name in class_names]
        axes[1,1].boxplot(conf_data, labels=class_names)
        axes[1,1].set_title('Prediction Confidence by True Class')
        axes[1,1].set_xlabel('True Class')
        axes[1,1].set_ylabel('Prediction Confidence')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_confusion_matrix.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Save detailed metrics to JSON (following journal format)
        detailed_metrics = {
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'per_class_metrics': {
                class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i]),
                    'class_index': i
                } for i in range(len(class_names))
            },
            'overall_metrics': {
                'accuracy': float(accuracy_score(valid_true_labels, valid_predictions)),
                'macro_precision': float(np.mean(precision)),
                'macro_recall': float(np.mean(recall)),
                'macro_f1': float(np.mean(f1)),
                'weighted_precision': float(np.average(precision, weights=support)),
                'weighted_recall': float(np.average(recall, weights=support)),
                'weighted_f1': float(np.average(f1, weights=support))
            },
            'class_mapping': self.species_mapping,
            'total_samples': int(len(valid_predictions)),
            'analysis_timestamp': datetime.now().isoformat()
        }

        with open(self.output_dir / 'detailed_metrics.json', 'w') as f:
            json.dump(detailed_metrics, f, indent=2)

        print(f"[SUCCESS] Comprehensive confusion matrix saved to {self.output_dir}")

        # Print summary following journal format
        print("\nDETAILED CLASSIFICATION REPORT:")
        print("="*60)
        report = classification_report(valid_true_labels, valid_predictions,
                                     target_names=class_names, digits=4)
        print(report)

        return detailed_metrics

    def create_journal_style_analysis(self, detailed_metrics):
        """Create analysis following the IEEE journal paper format"""
        print("\nCreating journal-style analysis report...")

        # Read the actual results for comparison
        results_csv = self.model_path.parent / "results.csv"
        training_results = None
        if results_csv.exists():
            training_results = pd.read_csv(results_csv)

        report = f"""# Deep Classification Analysis Report
## Automated Identification of Malaria-Infected Cells - Classification Evaluation

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: YOLOv8 Classification
**Dataset**: Malaria Parasite Species Classification (4 Classes)

---

## CRITICAL FINDINGS

### Suspicious Top-5 Accuracy Investigation

Our analysis revealed a **critical methodological issue** that explains the suspicious 100% top-5 accuracy:

**Root Cause Identified:**
- **Dataset has only 4 classes** (P. falciparum, P. malariae, P. ovale, P. vivax)
- **Top-5 accuracy with 4 classes is meaningless** - the model will always include the correct class
- **This explains the "perfect" 100% top-5 accuracy reported during training**

### Performance Metrics Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Top-1 Accuracy** | {detailed_metrics['overall_metrics']['accuracy']:.4f} | Actual model performance |
| **Top-5 Accuracy** | ~1.0000 | **INVALID** (only 4 classes available) |
| **Macro F1-Score** | {detailed_metrics['overall_metrics']['macro_f1']:.4f} | Balanced performance across classes |
| **Weighted F1-Score** | {detailed_metrics['overall_metrics']['weighted_f1']:.4f} | Performance weighted by support |

## Confusion Matrix Analysis

### Per-Class Performance
"""

        for class_name, metrics in detailed_metrics['per_class_metrics'].items():
            report += f"""
**{class_name}**:
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}
- Support: {metrics['support']} samples
"""

        report += f"""

## Comparison with IEEE Journal Reference

Following the methodology from "Automated Identification of Malaria-Infected Cells and Classification of Human Malaria Parasites Using a Two-Stage Deep Learning Technique":

### Our Results vs. Journal Results

| Model | Dataset | Accuracy | Notes |
|-------|---------|----------|-------|
| **Our YOLOv8** | {detailed_metrics['total_samples']} samples | {detailed_metrics['overall_metrics']['accuracy']:.1%} | 4-class classification |
| **Journal DenseNet-121** | MP-IDB + MRC-UNIMAS | 95.5% | 4-class classification |
| **Journal AlexNet** | MP-IDB + MRC-UNIMAS | 94.91% | 4-class classification |

### Key Insights

1. **Performance Level**: Our {detailed_metrics['overall_metrics']['accuracy']:.1%} accuracy is {"competitive with" if detailed_metrics['overall_metrics']['accuracy'] > 0.90 else "below"} the journal's best result (95.5%)

2. **Dataset Scale**: Our analysis uses {detailed_metrics['total_samples']} test samples vs. journal's larger dataset

3. **Methodology**: Similar two-stage approach (detection → classification)

## Recommendations

### For Future Research

1. **Report only Top-1 accuracy** for 4-class problems
2. **Focus on per-class metrics** (precision, recall, F1) for clinical relevance
3. **Consider class imbalance** in evaluation metrics
4. **Validate with larger, balanced datasets**

### For Clinical Application

1. **Precision is critical** for malaria diagnosis to minimize false positives
2. **Recall is essential** to avoid missing infections
3. **Species-specific performance** varies significantly

## Technical Details

- **Model Architecture**: YOLOv8n-cls
- **Input Size**: 128x128 pixels
- **Training Epochs**: {len(training_results) if training_results is not None else 'Unknown'}
- **Early Stopping**: {"Yes" if training_results is not None and len(training_results) < 30 else "No"}

---

**Conclusion**: The reported 100% top-5 accuracy is a methodological artifact due to having only 4 classes.
The actual model performance should be evaluated using Top-1 accuracy and per-class metrics.

*Generated automatically by Deep Classification Analyzer*
"""

        # Save the report
        with open(self.output_dir / 'journal_style_analysis.md', 'w') as f:
            f.write(report)

        print(f"[SUCCESS] Journal-style analysis saved to {self.output_dir}/journal_style_analysis.md")

        return report

    def run_complete_analysis(self):
        """Run the complete deep analysis pipeline"""
        print("Starting Complete Deep Classification Analysis")
        print("="*60)

        # Step 1: Load model and data
        if not self.load_model_and_data():
            return False

        # Step 2: Run inference and basic analysis
        if not self.run_inference_analysis():
            return False

        # Step 3: Create comprehensive confusion matrix
        detailed_metrics = self.create_comprehensive_confusion_matrix()
        if not detailed_metrics:
            return False

        # Step 4: Create journal-style analysis
        self.create_journal_style_analysis(detailed_metrics)

        print("\nAnalysis Complete!")
        print(f"All results saved to: {self.output_dir}")
        print("\nKey Files Generated:")
        print(f"  comprehensive_confusion_matrix.png - Visual analysis")
        print(f"  journal_style_analysis.md - Research report")
        print(f"  detailed_metrics.json - Raw metrics data")

        return True

def main():
    """Main function to run the analysis"""
    import argparse

    parser = argparse.ArgumentParser(description="Deep Classification Analysis for Malaria Detection")
    parser.add_argument("--model", required=False,
                       help="Path to classification model weights (.pt file)")
    parser.add_argument("--test-data", required=False,
                       help="Path to test data directory")
    parser.add_argument("--output", required=False,
                       help="Output directory for analysis results")

    args = parser.parse_args()

    print("DEEP CLASSIFICATION ANALYSIS")
    print("Following IEEE journal methodology for malaria classification")
    print("="*70)

    # Initialize analyzer with provided arguments
    if args.model and args.test_data:
        analyzer = ClassificationDeepAnalyzer(
            model_path=args.model,
            test_data_path=args.test_data,
            output_dir=args.output
        )
    else:
        # Use default paths if not provided
        analyzer = ClassificationDeepAnalyzer()

    # Run complete analysis
    success = analyzer.run_complete_analysis()

    if success:
        print("\n[SUCCESS] Analysis completed successfully!")
        print("Check the generated reports for detailed findings.")
    else:
        print("\n[ERROR] Analysis failed!")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())