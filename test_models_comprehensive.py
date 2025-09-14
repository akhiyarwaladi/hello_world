#!/usr/bin/env python3
"""
Comprehensive Model Testing with Integrated Result Logging
Tests trained models and saves detailed predictions, confusion matrices, and performance metrics
"""

import sys
import os
sys.path.append('.')

from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from utils.experiment_logger import ExperimentLogger

class ComprehensiveModelTester:
    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.results_dir / "predictions").mkdir(exist_ok=True)
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        (self.results_dir / "reports").mkdir(exist_ok=True)

    def test_detection_model(self, model_path: str, test_dataset: str, experiment_name: str = None):
        """Comprehensive testing of detection models"""

        if experiment_name is None:
            experiment_name = f"test_detection_{Path(model_path).stem}_{int(datetime.now().timestamp())}"

        # Initialize logger
        logger = ExperimentLogger(experiment_name, "detection_test", str(self.results_dir / "logs"))

        try:
            # Load model
            model = YOLO(model_path)
            logger.log_model_config(
                model_name=Path(model_path).stem,
                model_path=model_path,
                model_type="detection"
            )

            # Get test dataset info
            test_images = self._get_test_images(test_dataset)
            logger.log_dataset_info(
                dataset_path=test_dataset,
                num_classes=1,  # Malaria detection
                total_test_images=len(test_images)
            )

            print(f"🔍 Testing detection model: {Path(model_path).name}")
            print(f"📊 Test images: {len(test_images)}")

            # Run inference on all test images
            all_predictions = []
            all_ground_truths = []

            for i, image_path in enumerate(test_images):
                if i % 50 == 0:
                    print(f"Progress: {i}/{len(test_images)}")

                # Run prediction
                results = model(image_path)

                # Extract predictions
                pred_data = self._extract_detection_predictions(results[0], image_path)
                all_predictions.extend(pred_data["detections"])

                # Load ground truth if available
                gt_data = self._load_detection_ground_truth(image_path)
                if gt_data:
                    all_ground_truths.extend(gt_data)

            # Calculate metrics
            metrics = self._calculate_detection_metrics(all_predictions, all_ground_truths)

            # Log results
            logger.log_test_results(
                test_predictions=all_predictions,
                test_metrics=metrics
            )

            # Save detailed predictions
            self._save_detection_predictions(all_predictions, experiment_name)

            # Create visualizations
            self._create_detection_visualizations(all_predictions, metrics, experiment_name)

            # Final results
            logger.log_final_results(metrics)

            print(f"✅ Detection testing completed: {experiment_name}")
            return logger.get_summary()

        except Exception as e:
            logger.log_error(str(e), "detection_test")
            print(f"❌ Detection testing failed: {e}")
            return None

    def test_classification_model(self, model_path: str, test_dataset: str, class_names: list, experiment_name: str = None):
        """Comprehensive testing of classification models"""

        if experiment_name is None:
            experiment_name = f"test_classification_{Path(model_path).stem}_{int(datetime.now().timestamp())}"

        # Initialize logger
        logger = ExperimentLogger(experiment_name, "classification_test", str(self.results_dir / "logs"))

        try:
            # Load model
            model = YOLO(model_path)
            logger.log_model_config(
                model_name=Path(model_path).stem,
                model_path=model_path,
                model_type="classification",
                num_classes=len(class_names),
                class_names=class_names
            )

            # Get test dataset
            test_data = self._get_classification_test_data(test_dataset, class_names)
            logger.log_dataset_info(
                dataset_path=test_dataset,
                num_classes=len(class_names),
                total_test_images=len(test_data["images"]),
                class_distribution=test_data["class_counts"]
            )

            print(f"🔍 Testing classification model: {Path(model_path).name}")
            print(f"📊 Test images: {len(test_data['images'])}")
            print(f"🏷️  Classes: {class_names}")

            # Run inference
            all_predictions = []
            all_true_labels = []

            for i, (image_path, true_label) in enumerate(zip(test_data["images"], test_data["labels"])):
                if i % 100 == 0:
                    print(f"Progress: {i}/{len(test_data['images'])}")

                # Run prediction
                results = model(image_path)

                # Extract prediction
                pred_data = self._extract_classification_prediction(results[0], image_path, true_label, class_names)
                all_predictions.append(pred_data)
                all_true_labels.append(true_label)

            # Calculate metrics
            metrics = self._calculate_classification_metrics(all_predictions, all_true_labels, class_names)

            # Log results
            logger.log_test_results(
                test_predictions=all_predictions,
                test_metrics=metrics
            )

            # Save detailed results
            self._save_classification_results(all_predictions, metrics, class_names, experiment_name)

            # Create visualizations
            self._create_classification_visualizations(all_predictions, all_true_labels, metrics, class_names, experiment_name)

            # Final results
            logger.log_final_results(metrics)

            print(f"✅ Classification testing completed: {experiment_name}")
            return logger.get_summary()

        except Exception as e:
            logger.log_error(str(e), "classification_test")
            print(f"❌ Classification testing failed: {e}")
            return None

    def _get_test_images(self, test_dataset: str) -> list:
        """Get list of test images"""
        dataset_path = Path(test_dataset)

        if dataset_path.is_file() and dataset_path.suffix == '.yaml':
            # YOLO dataset
            import yaml
            with open(dataset_path, 'r') as f:
                config = yaml.safe_load(f)

            test_dir = dataset_path.parent / 'test' / 'images'
            if test_dir.exists():
                return list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))

        elif dataset_path.is_dir():
            # Directory of images
            return list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))

        return []

    def _get_classification_test_data(self, test_dataset: str, class_names: list) -> dict:
        """Get classification test data"""
        dataset_path = Path(test_dataset)

        images = []
        labels = []
        class_counts = {name: 0 for name in class_names}

        if dataset_path.is_dir():
            # Assume class subdirectories
            for class_name in class_names:
                class_dir = dataset_path / class_name
                if class_dir.exists():
                    class_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                    images.extend(class_images)
                    labels.extend([class_name] * len(class_images))
                    class_counts[class_name] = len(class_images)

        return {
            "images": images,
            "labels": labels,
            "class_counts": class_counts
        }

    def _extract_detection_predictions(self, result, image_path: str) -> dict:
        """Extract predictions from detection result"""
        detections = []

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                detection = {
                    "image_path": str(image_path),
                    "detection_id": i,
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": float(box.conf[0].cpu().numpy()),
                    "class_id": int(box.cls[0].cpu().numpy()),
                    "timestamp": datetime.now().isoformat()
                }
                detections.append(detection)

        return {"detections": detections}

    def _load_detection_ground_truth(self, image_path: str) -> list:
        """Load ground truth annotations for detection"""
        # Try to find corresponding label file
        image_path = Path(image_path)
        label_path = image_path.parent.parent / 'labels' / f"{image_path.stem}.txt"

        if label_path.exists():
            ground_truths = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        ground_truths.append({
                            "class_id": int(parts[0]),
                            "bbox_normalized": [float(x) for x in parts[1:5]]
                        })
            return ground_truths

        return []

    def _extract_classification_prediction(self, result, image_path: str, true_label: str, class_names: list) -> dict:
        """Extract prediction from classification result"""

        # Get top prediction
        probs = result.probs
        predicted_class_id = probs.top1
        predicted_class_name = class_names[predicted_class_id]
        confidence = float(probs.top1conf)

        # Get all class probabilities
        all_probs = {class_names[i]: float(prob) for i, prob in enumerate(probs.data)}

        return {
            "image_path": str(image_path),
            "true_label": true_label,
            "predicted_label": predicted_class_name,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "correct": predicted_class_name == true_label,
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_detection_metrics(self, predictions: list, ground_truths: list) -> dict:
        """Calculate detection metrics"""
        if not predictions:
            return {"error": "No predictions to evaluate"}

        # Basic statistics
        metrics = {
            "total_predictions": len(predictions),
            "total_ground_truths": len(ground_truths),
            "avg_confidence": np.mean([p["confidence"] for p in predictions]),
            "confidence_std": np.std([p["confidence"] for p in predictions]),
            "high_confidence_predictions": len([p for p in predictions if p["confidence"] > 0.5])
        }

        # If ground truth available, calculate precision/recall
        if ground_truths:
            metrics.update(self._calculate_detection_ap(predictions, ground_truths))

        return metrics

    def _calculate_detection_ap(self, predictions: list, ground_truths: list) -> dict:
        """Calculate Average Precision for detection"""
        # Simplified AP calculation
        # In practice, you'd want to implement proper IoU matching

        return {
            "estimated_precision": 0.0,  # Placeholder
            "estimated_recall": 0.0,     # Placeholder
            "note": "Full AP calculation requires IoU matching implementation"
        }

    def _calculate_classification_metrics(self, predictions: list, true_labels: list, class_names: list) -> dict:
        """Calculate classification metrics"""

        pred_labels = [p["predicted_label"] for p in predictions]
        confidences = [p["confidence"] for p in predictions]

        # Basic accuracy
        correct = sum(1 for p in predictions if p["correct"])
        accuracy = correct / len(predictions)

        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score

        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, labels=class_names, average=None, zero_division=0
        )

        # Create per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }

        # Overall metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='macro', zero_division=0
        )

        return {
            "accuracy": accuracy,
            "total_samples": len(predictions),
            "correct_predictions": correct,
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "avg_confidence": np.mean(confidences),
            "confidence_std": np.std(confidences),
            "per_class_metrics": per_class_metrics
        }

    def _save_detection_predictions(self, predictions: list, experiment_name: str):
        """Save detection predictions to file"""

        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(predictions)

        # Save as CSV
        csv_path = self.results_dir / "predictions" / f"{experiment_name}_detections.csv"
        df.to_csv(csv_path, index=False)

        # Save as JSON with full details
        json_path = self.results_dir / "predictions" / f"{experiment_name}_detections.json"
        with open(json_path, 'w') as f:
            json.dump(predictions, f, indent=2)

    def _save_classification_results(self, predictions: list, metrics: dict, class_names: list, experiment_name: str):
        """Save classification results"""

        # Save predictions
        df_pred = pd.DataFrame(predictions)
        pred_path = self.results_dir / "predictions" / f"{experiment_name}_classifications.csv"
        df_pred.to_csv(pred_path, index=False)

        # Save detailed metrics
        metrics_path = self.results_dir / "reports" / f"{experiment_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create classification report
        true_labels = [p["true_label"] for p in predictions]
        pred_labels = [p["predicted_label"] for p in predictions]

        report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)

        report_path = self.results_dir / "reports" / f"{experiment_name}_sklearn_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    def _create_detection_visualizations(self, predictions: list, metrics: dict, experiment_name: str):
        """Create detection visualizations"""

        if not predictions:
            return

        # Confidence distribution
        confidences = [p["confidence"] for p in predictions]

        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of Detections')
        plt.title(f'Detection Confidence Distribution - {experiment_name}')
        plt.grid(True, alpha=0.3)

        # Add statistics
        plt.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
        plt.axvline(np.median(confidences), color='green', linestyle='--', label=f'Median: {np.median(confidences):.3f}')
        plt.legend()

        viz_path = self.results_dir / "visualizations" / f"{experiment_name}_confidence_dist.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_classification_visualizations(self, predictions: list, true_labels: list, metrics: dict, class_names: list, experiment_name: str):
        """Create classification visualizations"""

        pred_labels = [p["predicted_label"] for p in predictions]

        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=class_names)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {experiment_name}')

        viz_path = self.results_dir / "visualizations" / f"{experiment_name}_confusion_matrix.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Confidence distribution per class
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, class_name in enumerate(class_names[:4]):  # Show first 4 classes
            class_predictions = [p for p in predictions if p["true_label"] == class_name]
            if class_predictions:
                confidences = [p["confidence"] for p in class_predictions]

                axes[i].hist(confidences, bins=15, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel('Confidence')
                axes[i].set_ylabel('Count')
                axes[i].set_title(f'{class_name} - Confidence Distribution')
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        viz_path = self.results_dir / "visualizations" / f"{experiment_name}_confidence_by_class.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main testing function"""
    print("🧪 COMPREHENSIVE MODEL TESTING")
    print("=" * 50)

    tester = ComprehensiveModelTester()

    # Test completed models
    test_configs = [
        {
            "type": "detection",
            "model_path": "results/pipeline_final/multispecies_detection_final/weights/best.pt",
            "test_dataset": "data/detection_multispecies/dataset.yaml",
            "experiment_name": "multispecies_detection_test"
        },
        {
            "type": "classification",
            "model_path": "results/pipeline_final/multispecies_classification/weights/best.pt",
            "test_dataset": "data/classification_multispecies",
            "class_names": ["falciparum", "vivax", "malariae", "ovale"],
            "experiment_name": "multispecies_classification_test"
        }
    ]

    results = []

    for config in test_configs:
        model_path = Path(config["model_path"])
        if model_path.exists():
            print(f"\n🔍 Testing {config['type']} model: {model_path.name}")

            if config["type"] == "detection":
                result = tester.test_detection_model(
                    str(model_path),
                    config["test_dataset"],
                    config["experiment_name"]
                )
            else:
                result = tester.test_classification_model(
                    str(model_path),
                    config["test_dataset"],
                    config["class_names"],
                    config["experiment_name"]
                )

            if result:
                results.append(result)
        else:
            print(f"⚠️  Model not found: {model_path}")

    print(f"\n✅ Testing completed! Results for {len(results)} models saved.")
    print(f"📁 Check results in: {tester.results_dir}")

if __name__ == "__main__":
    main()