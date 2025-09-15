#!/usr/bin/env python3
"""
Hybrid Detection + Classification Pipeline
Combines different models for detection and classification
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.results_manager import ResultsManager

class HybridMalariaDetector:
    """
    Hybrid detector that combines:
    - Detection Model: YOLOv8/YOLOv11/RT-DETR for parasite detection
    - Classification Model: Custom models for species classification
    """

    def __init__(self, detection_model_path, classification_model_path):
        self.detection_model = YOLO(detection_model_path)

        # For now, use YOLO classification, but this can be replaced
        # with any PyTorch model (ResNet, EfficientNet, etc.)
        if classification_model_path.endswith('.pt') and 'yolo' in classification_model_path.lower():
            self.classification_model = YOLO(classification_model_path)
            self.is_yolo_classifier = True
        else:
            # Placeholder for custom models
            self.classification_model = self.load_custom_classifier(classification_model_path)
            self.is_yolo_classifier = False

    def load_custom_classifier(self, model_path):
        """Load custom PyTorch classification model"""
        # This can be extended to load ResNet, EfficientNet, etc.
        print(f"⚠️  Custom model loading not implemented yet: {model_path}")
        print("💡 Currently supporting YOLO classification models only")
        return None

    def detect_and_classify(self, image_path, confidence=0.5):
        """
        Full pipeline: detect parasites then classify species

        Args:
            image_path: Path to microscopy image
            confidence: Detection confidence threshold

        Returns:
            List of detections with classifications
        """
        # Step 1: Detection
        print(f"🔍 Detecting parasites in {image_path}")
        detection_results = self.detection_model.predict(
            image_path,
            conf=confidence,
            verbose=False
        )

        results = []

        for result in detection_results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                # Step 2: Classify each detected parasite
                image = Image.open(image_path)

                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    # Crop detected region
                    x1, y1, x2, y2 = map(int, box)
                    crop = image.crop((x1, y1, x2, y2))

                    # Classify species
                    if self.is_yolo_classifier and self.classification_model:
                        # Save temp crop for YOLO classifier
                        temp_crop_path = f"temp_crop_{i}.jpg"
                        crop.save(temp_crop_path)

                        class_result = self.classification_model.predict(
                            temp_crop_path,
                            verbose=False
                        )

                        if class_result[0].probs is not None:
                            top_class = class_result[0].probs.top1
                            class_conf = class_result[0].probs.top1conf.item()
                            species = self.classification_model.names[top_class]
                        else:
                            species = "unknown"
                            class_conf = 0.0

                        # Clean up temp file
                        os.remove(temp_crop_path)
                    else:
                        species = "not_classified"
                        class_conf = 0.0

                    results.append({
                        'detection_box': box,
                        'detection_confidence': conf,
                        'species': species,
                        'classification_confidence': class_conf
                    })

        return results

def train_hybrid_models(detection_model_type, classification_model_type):
    """Train detection and classification models separately"""

    results_manager = ResultsManager()
    experiment_path = results_manager.get_experiment_path(
        experiment_type="training",
        model_name="hybrid_pipeline",
        experiment_name=f"{detection_model_type}_{classification_model_type}"
    )

    print("🔄 Training Hybrid Pipeline:")
    print(f"   Detection: {detection_model_type}")
    print(f"   Classification: {classification_model_type}")
    print(f"   Results: {experiment_path}")

    # Train detection model
    if detection_model_type == "yolov8":
        detection_script = "scripts/07_train_yolo_detection.py"
    elif detection_model_type == "yolov11":
        detection_script = "scripts/08_train_yolo11_detection.py"
    elif detection_model_type == "rtdetr":
        detection_script = "scripts/09_train_rtdetr_detection.py"
    else:
        raise ValueError(f"Unsupported detection model: {detection_model_type}")

    # Train classification model
    if classification_model_type in ["yolov8", "yolov11"]:
        if classification_model_type == "yolov8":
            class_script = "scripts/11_train_classification_crops.py"
            class_model = "yolov8n-cls.pt"
        else:  # yolov11
            class_script = "yolo classify train"
            class_model = "yolo11n-cls.pt"
    else:
        print(f"⚠️  Custom classification model {classification_model_type} not implemented yet")
        print("💡 Use yolov8 or yolov11 for now")
        return

    print(f"\n🎯 Step 1: Training {detection_model_type} detection model")
    print(f"Command: python {detection_script}")

    print(f"\n🎯 Step 2: Training {classification_model_type} classification model")
    if "yolo" in class_script:
        print(f"Command: {class_script} data=data/classification_multispecies model={class_model}")
    else:
        print(f"Command: python {class_script} --model {class_model}")

def main():
    parser = argparse.ArgumentParser(description="Hybrid Detection + Classification Pipeline")
    parser.add_argument("mode", choices=["train", "inference"],
                       help="Mode: train models or run inference")

    # Training arguments
    parser.add_argument("--detection_model", default="yolov11",
                       choices=["yolov8", "yolov11", "rtdetr"],
                       help="Detection model type")
    parser.add_argument("--classification_model", default="yolov8",
                       choices=["yolov8", "yolov11", "resnet50", "efficientnet"],
                       help="Classification model type")

    # Inference arguments
    parser.add_argument("--detection_weights",
                       help="Path to trained detection model weights")
    parser.add_argument("--classification_weights",
                       help="Path to trained classification model weights")
    parser.add_argument("--image", help="Input image for inference")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Detection confidence threshold")

    args = parser.parse_args()

    if args.mode == "train":
        print("🚀 HYBRID MODEL TRAINING")
        print("=" * 50)
        train_hybrid_models(args.detection_model, args.classification_model)

    elif args.mode == "inference":
        if not all([args.detection_weights, args.classification_weights, args.image]):
            print("❌ For inference, provide: --detection_weights, --classification_weights, --image")
            return

        print("🔍 HYBRID INFERENCE")
        print("=" * 50)

        detector = HybridMalariaDetector(args.detection_weights, args.classification_weights)
        results = detector.detect_and_classify(args.image, args.confidence)

        print(f"\n📊 Results for {args.image}:")
        print(f"   Found {len(results)} parasites")

        for i, result in enumerate(results):
            print(f"\n   Parasite {i+1}:")
            print(f"     Detection confidence: {result['detection_confidence']:.3f}")
            print(f"     Species: {result['species']}")
            print(f"     Classification confidence: {result['classification_confidence']:.3f}")

if __name__ == "__main__":
    main()