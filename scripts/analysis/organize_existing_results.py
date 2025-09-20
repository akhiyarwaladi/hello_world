#!/usr/bin/env python3
"""
ORGANIZE EXISTING RESULTS: Create organized analysis for completed experiments
Mengorganisir hasil training yang sudah selesai tanpa perlu training ulang
"""

import os
import sys
import json
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

class ExistingResultsOrganizer:
    def __init__(self, experiment_pattern="multi_pipeline_20250920_131500"):
        self.experiment_pattern = experiment_pattern
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.organized_dir = Path(f"experiments_organized_{self.timestamp}")
        self.organized_dir.mkdir(exist_ok=True)

        print(f"🔄 ORGANIZING EXISTING EXPERIMENT RESULTS")
        print(f"📁 Looking for pattern: {experiment_pattern}")
        print(f"📂 Output directory: {self.organized_dir}")

    def find_completed_experiments(self):
        """Find all completed experiments matching the pattern"""
        experiments = {}

        # Model mappings
        model_info = {
            'yolo8': 'yolov8_detection',
            'yolo11': 'yolo11_detection',
            'yolo12': 'yolo12_detection',
            'rtdetr': 'rtdetr_detection'
        }

        for model_key, detection_model in model_info.items():
            # Find detection results
            det_pattern = f"{self.experiment_pattern}_{model_key}_det"
            det_path = Path(f"results/current_experiments/training/detection/{detection_model}/{det_pattern}")

            # Find classification results
            cls_pattern = f"{self.experiment_pattern}_{model_key}_cls"
            cls_path = Path(f"results/current_experiments/training/classification/yolov8_classification/{cls_pattern}")

            # Find crop data
            crop_pattern = f"data/crops_from_{model_key}_{det_pattern}"
            crop_path = Path(crop_pattern)

            if det_path.exists():
                experiments[model_key] = {
                    'detection_path': det_path,
                    'classification_path': cls_path if cls_path.exists() else None,
                    'crop_path': crop_path if crop_path.exists() else None,
                    'detection_model': detection_model,
                    'det_exp_name': det_pattern,
                    'cls_exp_name': cls_pattern
                }

                status = "✅ Complete" if (cls_path.exists() and crop_path.exists()) else "⚠️ Partial"
                print(f"   {status} {model_key.upper()}: Detection={det_path.exists()}, Classification={cls_path.exists()}, Crops={crop_path.exists()}")

        return experiments

    def create_organized_experiment(self, model_key, exp_data):
        """Create organized experiment structure for existing results"""
        print(f"\n📁 Organizing {model_key.upper()} results...")

        # Create experiment directory
        exp_dir = self.organized_dir / f"{model_key}_complete"
        exp_dir.mkdir(exist_ok=True)

        # Create subdirectories
        subdirs = {
            "01_detection_results": exp_data['detection_path'],
            "02_crop_data": exp_data['crop_path'],
            "03_classification_results": exp_data['classification_path'],
            "04_analysis": None,  # Will create analysis here
            "05_summary": None    # Will create summary here
        }

        for subdir_name, source_path in subdirs.items():
            subdir_path = exp_dir / subdir_name
            subdir_path.mkdir(exist_ok=True)

            if source_path and source_path.exists():
                # Create symlink to original data (saves space)
                link_path = subdir_path / "original_data"
                if not link_path.exists():
                    try:
                        link_path.symlink_to(source_path.absolute())
                        print(f"   🔗 Linked {subdir_name} → {source_path}")
                    except Exception as e:
                        print(f"   ⚠️ Could not create symlink for {subdir_name}: {e}")

        return exp_dir

    def run_iou_variation_analysis(self, model_key, exp_data, exp_dir):
        """Run IoU variation analysis at different thresholds (0.3, 0.5, 0.7)"""
        print(f"📊 Running IoU variation analysis for {model_key.upper()}...")

        analysis_dir = exp_dir / "04_analysis"
        iou_dir = analysis_dir / "iou_variation"
        iou_dir.mkdir(exist_ok=True)

        # Check if we have detection model
        det_model_path = None
        test_images_path = None

        if exp_data['detection_path'] and exp_data['detection_path'].exists():
            det_model_path = exp_data['detection_path'] / "weights" / "best.pt"

        # Find test images in the original dataset
        test_images_paths = [
            Path("data/integrated/yolo/test/images"),
            Path("data/integrated_dataset/test/images"),
            Path("data/test/images"),
            Path("data/dataset/test/images")
        ]

        for path in test_images_paths:
            if path.exists():
                test_images_path = path
                break

        if det_model_path and det_model_path.exists() and test_images_path and test_images_path.exists():
            print(f"   📊 Testing IoU thresholds: 0.3, 0.5, 0.7...")

            # Create IoU variation analysis script
            iou_script_content = f'''#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
sys.path.append('/home/akhiyarwaladi/hello_world')

try:
    from ultralytics import YOLO
    import pandas as pd

    # Load model
    model = YOLO("{det_model_path}")
    print(f"📦 Loaded model: {model_key.upper()}")

    # Test images directory
    test_dir = Path("{test_images_path}")
    print(f"🖼️ Test images: {{test_dir}}")

    # IoU thresholds to test (following reference paper)
    iou_thresholds = [0.3, 0.5, 0.7]

    results_summary = {{}}

    for iou_thresh in iou_thresholds:
        print(f"\\n🎯 Testing IoU threshold: {{iou_thresh}}")

        # Run validation with specific IoU threshold on TEST SET
        # IMPORTANT: Using split='test' to ensure independent evaluation
        # Note: YOLO uses iou parameter for NMS, but mAP calculation uses fixed thresholds
        metrics = model.val(
            data="data/integrated/yolo/data.yaml",
            split='test',  # Use test set, NOT validation set
            iou=iou_thresh,
            verbose=False,
            save=False
        )

        # Extract key metrics
        results_summary[f"iou_{{iou_thresh:.1f}}"] = {{
            "iou_threshold": iou_thresh,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "confidence_threshold": 0.25  # Default YOLO confidence
        }}

        print(f"   mAP@0.5: {{metrics.box.map50:.3f}}")
        print(f"   mAP@0.5:0.95: {{metrics.box.map:.3f}}")
        print(f"   Precision: {{metrics.box.mp:.3f}}")
        print(f"   Recall: {{metrics.box.mr:.3f}}")

    # Save results
    output_file = Path("{iou_dir}") / "iou_variation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Create comparison table
    comparison_data = []
    for iou_key, metrics in results_summary.items():
        comparison_data.append({{
            "IoU_Threshold": metrics["iou_threshold"],
            "mAP@0.5": f"{{metrics['map50']:.3f}}",
            "mAP@0.5:0.95": f"{{metrics['map50_95']:.3f}}",
            "Precision": f"{{metrics['precision']:.3f}}",
            "Recall": f"{{metrics['recall']:.3f}}"
        }})

    df = pd.DataFrame(comparison_data)
    csv_file = Path("{iou_dir}") / "iou_comparison_table.csv"
    df.to_csv(csv_file, index=False)

    # Create markdown report
    md_content = f"""# IoU Variation Analysis - {{model_key.upper()}}

## 📊 Performance at Different IoU Thresholds

Following the reference paper methodology, we test the detection model at different IoU thresholds:

| IoU Threshold | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------------|---------|--------------|-----------|--------|
"""

    for data in comparison_data:
        md_content += f"| {{data['IoU_Threshold']}} | {{data['mAP@0.5']}} | {{data['mAP@0.5:0.95']}} | {{data['Precision']}} | {{data['Recall']}} |\\n"

    md_content += f"""
## 📚 Reference Paper Comparison

**Paper Parameters:**
- IoU=0.7 for training
- Testing at IoU: 0.3, 0.5, 0.7
- YOLOv4: mAP@0.5=89% (source), 90% (cross-test)
- YOLOv5: mAP@0.5=96% (source), 59% (cross-test)

**Our Results (TEST SET - Independent Evaluation):**
- Model: {{model_key.upper()}}
- Best mAP@0.5: {{max(results_summary.values(), key=lambda x: x['map50'])['map50']:.3f}} at IoU={{max(results_summary.values(), key=lambda x: x['map50'])['iou_threshold']}}
- Detection confidence threshold: 0.25 (fixed)
- ⚠️ IMPORTANT: These are TEST SET results (not validation)

## 🔧 Analysis Notes

- **IoU Threshold**: Controls Non-Maximum Suppression (NMS) overlap tolerance
- **Confidence Threshold**: Fixed at 0.25 for detection filtering
- **Higher IoU**: More strict overlap requirement (fewer detections)
- **Lower IoU**: More lenient overlap (more detections, potential duplicates)

## 📁 Files Generated

- `iou_variation_results.json`: Raw metrics data
- `iou_comparison_table.csv`: Comparison table
- `iou_analysis_report.md`: This report

---
*Generated by IoU Variation Analysis*
"""

    md_file = Path("{iou_dir}") / "iou_analysis_report.md"
    with open(md_file, 'w') as f:
        f.write(md_content)

    print(f"\\n✅ IoU variation analysis completed!")
    print(f"📁 Results saved to: {{Path('{iou_dir}')}}")
    print(f"📊 Best mAP@0.5: {{max(results_summary.values(), key=lambda x: x['map50'])['map50']:.3f}} at IoU={{max(results_summary.values(), key=lambda x: x['map50'])['iou_threshold']}}")

except Exception as e:
    print(f"⚠️ IoU analysis failed: {{e}}")
    import traceback
    traceback.print_exc()
'''

            # Write and run IoU analysis script
            iou_script_path = exp_dir / "run_iou_analysis.py"
            with open(iou_script_path, 'w') as f:
                f.write(iou_script_content)
            iou_script_path.chmod(0o755)

            # Run IoU analysis
            import subprocess
            try:
                result = subprocess.run(["python3", str(iou_script_path)],
                                      capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    print(f"   ✅ IoU analysis completed for {model_key}")
                    return True
                else:
                    print(f"   ⚠️ IoU analysis had issues: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print(f"   ⚠️ IoU analysis timed out for {model_key}")
                return False
            except Exception as e:
                print(f"   ⚠️ IoU analysis failed: {e}")
                return False
        else:
            print(f"   ⚠️ Skipping IoU analysis - missing required files")
            print(f"      Detection model: {det_model_path.exists() if det_model_path else False}")
            print(f"      Test images: {test_images_path.exists() if test_images_path else False}")
            return False

    def run_analysis_for_experiment(self, model_key, exp_data, exp_dir):
        """Run analysis for this specific experiment"""
        print(f"🔬 Running analysis for {model_key.upper()}...")

        analysis_dir = exp_dir / "04_analysis"

        # 1. Run IoU variation analysis first
        iou_success = self.run_iou_variation_analysis(model_key, exp_data, exp_dir)

        # 2. Run classification analysis if available
        cls_model_path = None
        test_data_path = None

        if exp_data['classification_path'] and exp_data['classification_path'].exists():
            cls_model_path = exp_data['classification_path'] / "weights" / "best.pt"

        if exp_data['crop_path'] and exp_data['crop_path'].exists():
            test_data_path = exp_data['crop_path'] / "yolo_classification" / "test"

        if cls_model_path and cls_model_path.exists() and test_data_path and test_data_path.exists():
            print(f"   📊 Running confusion matrix analysis...")

            # Create custom analysis script
            analysis_script_content = f'''#!/usr/bin/env python3
import os
import sys
sys.path.append('/home/akhiyarwaladi/hello_world')

try:
    from scripts.analysis.classification_deep_analysis import DeepClassificationAnalyzer

    analyzer = DeepClassificationAnalyzer(
        model_path="{cls_model_path}",
        test_data_path="{test_data_path}",
        output_dir="{analysis_dir}"
    )

    print("🔬 Running comprehensive analysis...")
    result = analyzer.run_complete_analysis()
    print("✅ Classification analysis completed successfully!")

except Exception as e:
    print(f"⚠️ Classification analysis failed: {{e}}")
    import traceback
    traceback.print_exc()
'''

            # Write and run analysis script
            script_path = exp_dir / "run_classification_analysis.py"
            with open(script_path, 'w') as f:
                f.write(analysis_script_content)
            script_path.chmod(0o755)

            # Run analysis
            import subprocess
            try:
                result = subprocess.run(["python3", str(script_path)],
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print(f"   ✅ Classification analysis completed for {model_key}")
                else:
                    print(f"   ⚠️ Classification analysis had issues: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"   ⚠️ Classification analysis timed out for {model_key}")
            except Exception as e:
                print(f"   ⚠️ Classification analysis failed: {e}")
        else:
            print(f"   ⚠️ Skipping classification analysis - missing required files")
            print(f"      Model: {cls_model_path.exists() if cls_model_path else False}")
            print(f"      Test data: {test_data_path.exists() if test_data_path else False}")

        return iou_success

    def create_experiment_summary(self, model_key, exp_data, exp_dir):
        """Create comprehensive experiment summary"""
        print(f"📋 Creating summary for {model_key.upper()}...")

        summary_dir = exp_dir / "05_summary"

        try:
            summary_data = {
                "experiment_info": {
                    "model": model_key.upper(),
                    "detection_experiment": exp_data['det_exp_name'],
                    "classification_experiment": exp_data['cls_exp_name'],
                    "organized_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "original_experiment_pattern": self.experiment_pattern
                }
            }

            # Get detection results
            if exp_data['detection_path'] and exp_data['detection_path'].exists():
                det_results_file = exp_data['detection_path'] / "results.csv"
                if det_results_file.exists():
                    det_df = pd.read_csv(det_results_file)
                    final_det = det_df.iloc[-1]
                    summary_data["detection"] = {
                        "epochs": len(det_df),
                        "mAP50": float(final_det.get('metrics/mAP50(B)', 0)),
                        "mAP50_95": float(final_det.get('metrics/mAP50-95(B)', 0)),
                        "precision": float(final_det.get('metrics/precision(B)', 0)),
                        "recall": float(final_det.get('metrics/recall(B)', 0)),
                        "training_time_sec": float(final_det.get('time', 0))
                    }

            # Get classification results
            if exp_data['classification_path'] and exp_data['classification_path'].exists():
                cls_results_file = exp_data['classification_path'] / "results.csv"
                if cls_results_file.exists():
                    cls_df = pd.read_csv(cls_results_file)
                    final_cls = cls_df.iloc[-1]
                    summary_data["classification"] = {
                        "epochs": len(cls_df),
                        "top1_accuracy": float(final_cls.get('metrics/accuracy_top1', 0)),
                        "top5_accuracy": float(final_cls.get('metrics/accuracy_top5', 0)),
                        "training_time_sec": float(final_cls.get('time', 0))
                    }

            # Count crops if available
            if exp_data['crop_path'] and exp_data['crop_path'].exists():
                crop_counts = {}
                yolo_cls_path = exp_data['crop_path'] / "yolo_classification"
                if yolo_cls_path.exists():
                    for split in ["train", "val", "test"]:
                        split_path = yolo_cls_path / split
                        if split_path.exists():
                            for class_dir in split_path.iterdir():
                                if class_dir.is_dir():
                                    count = len(list(class_dir.glob("*.jpg")))
                                    crop_counts[f"{split}_{class_dir.name}"] = count
                summary_data["crops"] = crop_counts

            # Save JSON summary
            with open(summary_dir / "experiment_summary.json", 'w') as f:
                json.dump(summary_data, f, indent=2)

            # Create markdown summary
            total_time = (summary_data.get("detection", {}).get("training_time_sec", 0) +
                         summary_data.get("classification", {}).get("training_time_sec", 0)) / 60

            # Check for IoU analysis results
            iou_results_file = exp_dir / "04_analysis" / "iou_variation" / "iou_variation_results.json"
            iou_analysis_available = iou_results_file.exists()
            if iou_analysis_available:
                try:
                    with open(iou_results_file, 'r') as f:
                        iou_data = json.load(f)
                        summary_data["iou_analysis"] = iou_data
                except Exception as e:
                    print(f"   ⚠️ Could not load IoU analysis: {e}")

            md_content = f"""# {model_key.upper()} Complete Pipeline Results (Organized)

**Original Experiment**: {self.experiment_pattern}
**Organized**: {summary_data['experiment_info']['organized_timestamp']}

## 📊 Performance Summary

"""

            if 'detection' in summary_data:
                det = summary_data['detection']
                md_content += f"""### Detection Performance
- **mAP50**: {det.get('mAP50', 'N/A'):.3f}
- **mAP50-95**: {det.get('mAP50_95', 'N/A'):.3f}
- **Precision**: {det.get('precision', 'N/A'):.3f}
- **Recall**: {det.get('recall', 'N/A'):.3f}
- **Training Time**: {det.get('training_time_sec', 0)/60:.1f} minutes
- **Epochs**: {det.get('epochs', 'N/A')}

"""

            # Add IoU Analysis section if available
            if 'iou_analysis' in summary_data:
                md_content += f"""### IoU Variation Analysis (NMS Thresholds)

| IoU Threshold | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------------|---------|--------------|-----------|--------|
"""
                for iou_key, metrics in summary_data['iou_analysis'].items():
                    iou_val = metrics['iou_threshold']
                    map50 = metrics['map50']
                    map50_95 = metrics['map50_95']
                    precision = metrics['precision']
                    recall = metrics['recall']
                    md_content += f"| {iou_val} | {map50:.3f} | {map50_95:.3f} | {precision:.3f} | {recall:.3f} |\n"

                # Find best performing IoU
                best_iou = max(summary_data['iou_analysis'].values(), key=lambda x: x['map50'])
                md_content += f"""
**Best Performance**: mAP@0.5={best_iou['map50']:.3f} at IoU={best_iou['iou_threshold']}

"""

            if 'classification' in summary_data:
                cls = summary_data['classification']
                md_content += f"""### Classification Performance
- **Top-1 Accuracy**: {cls.get('top1_accuracy', 'N/A'):.3f}
- **Top-5 Accuracy**: {cls.get('top5_accuracy', 'N/A'):.3f} (Expected 1.0 for 4-class)
- **Training Time**: {cls.get('training_time_sec', 0)/60:.1f} minutes
- **Epochs**: {cls.get('epochs', 'N/A')}

"""

            md_content += f"""### Total Training Time: {total_time:.1f} minutes

## 📁 Organized Structure
```
{exp_dir.name}/
├── 01_detection_results/     # Detection training results (symlinked)
├── 02_crop_data/            # Generated crop dataset (symlinked)
├── 03_classification_results/ # Classification results (symlinked)
├── 04_analysis/             # Confusion matrix & deep analysis
└── 05_summary/              # This summary
```

## 🔗 Original Results Locations
"""

            if exp_data['detection_path']:
                md_content += f"- **Detection**: {exp_data['detection_path']}\n"
            if exp_data['classification_path']:
                md_content += f"- **Classification**: {exp_data['classification_path']}\n"
            if exp_data['crop_path']:
                md_content += f"- **Crops**: {exp_data['crop_path']}\n"

            md_content += f"""
---
*Organized by Existing Results Organizer*
"""

            with open(summary_dir / "experiment_summary.md", 'w') as f:
                f.write(md_content)

            print(f"   📋 Summary created: {summary_dir}/experiment_summary.md")

        except Exception as e:
            print(f"   ⚠️ Could not create summary: {e}")

    def organize_all_experiments(self):
        """Main function to organize all found experiments"""
        print(f"\n🔍 Scanning for completed experiments...")

        experiments = self.find_completed_experiments()

        if not experiments:
            print("❌ No completed experiments found!")
            return

        print(f"\n✅ Found {len(experiments)} experiments to organize")

        organized_count = 0

        for model_key, exp_data in experiments.items():
            try:
                print(f"\n{'='*60}")
                print(f"🔄 ORGANIZING {model_key.upper()}")
                print(f"{'='*60}")

                # Create organized structure
                exp_dir = self.create_organized_experiment(model_key, exp_data)

                # Run analysis
                self.run_analysis_for_experiment(model_key, exp_data, exp_dir)

                # Create summary
                self.create_experiment_summary(model_key, exp_data, exp_dir)

                organized_count += 1
                print(f"✅ {model_key.upper()} organized successfully!")
                print(f"📁 Location: {exp_dir}")

            except Exception as e:
                print(f"❌ Failed to organize {model_key}: {e}")

        # Final summary
        print(f"\n{'🎉'*30}")
        print(f"🎉 ORGANIZATION COMPLETE!")
        print(f"{'🎉'*30}")
        print(f"✅ Successfully organized: {organized_count}/{len(experiments)} experiments")
        print(f"📂 Main directory: {self.organized_dir}")

        return self.organized_dir

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Organize existing experiment results with analysis")
    parser.add_argument("--pattern", default="multi_pipeline_20250920_131500",
                       help="Experiment pattern to look for")

    args = parser.parse_args()

    organizer = ExistingResultsOrganizer(args.pattern)
    organized_dir = organizer.organize_all_experiments()

    return organized_dir

if __name__ == "__main__":
    main()