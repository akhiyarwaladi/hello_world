#!/usr/bin/env python3
import os
import sys
sys.path.append('/home/akhiyarwaladi/hello_world')

try:
    from scripts.analysis.classification_deep_analysis import DeepClassificationAnalyzer

    analyzer = DeepClassificationAnalyzer(
        model_path="results/current_experiments/training/classification/yolov8_classification/multi_pipeline_20250920_131500_yolo8_cls/weights/best.pt",
        test_data_path="data/crops_from_yolo8_multi_pipeline_20250920_131500_yolo8_det/yolo_classification/test",
        output_dir="experiments_organized_20250920_141008/yolo8_complete/04_analysis"
    )

    print("🔬 Running comprehensive analysis...")
    result = analyzer.run_complete_analysis()
    print("✅ Analysis completed successfully!")

except Exception as e:
    print(f"⚠️ Analysis failed: {e}")
    import traceback
    traceback.print_exc()
