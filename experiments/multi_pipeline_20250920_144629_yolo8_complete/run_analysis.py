#!/usr/bin/env python3
import os
import sys
import shutil
sys.path.append('/home/akhiyarwaladi/hello_world')

# Import the analysis class
try:
    from scripts.analysis.classification_deep_analysis import DeepClassificationAnalyzer

    analyzer = DeepClassificationAnalyzer(
        model_path="results/current_experiments/training/classification/yolov8_classification/multi_pipeline_20250920_144629_yolo8_cls/weights/best.pt",
        test_data_path="data/crops_from_yolo8_multi_pipeline_20250920_144629_yolo8_det/yolo_classification/test",
        output_dir="experiments/multi_pipeline_20250920_144629_yolo8_complete/analysis"
    )

    print("🔬 Running comprehensive analysis...")
    analyzer.run_complete_analysis()
    print("✅ Analysis completed successfully!")

except Exception as e:
    print(f"⚠️ Analysis failed: {e}")
