#!/usr/bin/env python3
"""
Verify Completeness of optA_20251016_200330 Experiment Results
Checks all expected outputs from main_pipeline.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def check_file_exists(path, description=""):
    """Check if file exists and return status"""
    exists = Path(path).exists()
    size = Path(path).stat().st_size if exists else 0
    return {
        'exists': exists,
        'path': str(path),
        'size': size,
        'description': description
    }

def analyze_experiment():
    """Analyze completeness of experiment results"""

    base_path = Path(r"C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251016_200330")

    report = {
        'experiment_name': 'optA_20251016_200330',
        'base_path': str(base_path),
        'datasets': {},
        'consolidated_analysis': {},
        'summary': {
            'total_files': 0,
            'total_size_mb': 0,
            'missing_files': [],
            'issues': []
        }
    }

    # Expected datasets
    datasets = ['iml_lifecycle', 'mp_idb_species', 'mp_idb_stages', 'md_2019_stages']

    # Expected models
    detection_models = ['yolo10', 'yolo11', 'yolo12']
    classification_models = [
        'densenet121_focal',
        'efficientnet_b0_focal',
        'efficientnet_b1_focal',
        'efficientnet_b2_focal',
        'resnet50_focal',
        'resnet101_focal'
    ]

    print("="*80)
    print(f"VERIFYING EXPERIMENT: {report['experiment_name']}")
    print("="*80)

    # Check each dataset
    for dataset in datasets:
        exp_path = base_path / "experiments" / f"experiment_{dataset}"

        print(f"\n{'='*80}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*80}")

        dataset_report = {
            'experiment_path': str(exp_path),
            'detection': {},
            'classification': {},
            'crops': {},
            'analysis': {},
            'visualizations': {},
            'summary_files': {},
            'file_count': 0,
            'total_size_mb': 0
        }

        if not exp_path.exists():
            print(f"❌ MISSING: Experiment folder not found")
            report['summary']['issues'].append(f"{dataset}: Experiment folder missing")
            continue

        # 1. Check Detection Models
        print(f"\n[1] DETECTION MODELS (3 expected)")
        for det_model in detection_models:
            det_path = exp_path / f"det_{det_model}"
            det_info = {
                'path': str(det_path),
                'exists': det_path.exists(),
                'files': {}
            }

            if det_path.exists():
                # Check expected files
                expected_files = {
                    'results.csv': 'Training metrics',
                    'args.yaml': 'Training arguments',
                    'weights/best.pt': 'Best model weights',
                    'weights/last.pt': 'Last checkpoint',
                    'results.png': 'Training curves',
                    'confusion_matrix.png': 'Confusion matrix',
                    'BoxPR_curve.png': 'Precision-Recall curve'
                }

                for file, desc in expected_files.items():
                    file_path = det_path / file
                    det_info['files'][file] = check_file_exists(file_path, desc)
                    if file_path.exists():
                        dataset_report['file_count'] += 1
                        dataset_report['total_size_mb'] += file_path.stat().st_size / (1024**2)

                # Count all files
                all_files = list(det_path.rglob("*"))
                file_count = len([f for f in all_files if f.is_file()])
                det_info['total_files'] = file_count

                missing = [f for f, info in det_info['files'].items() if not info['exists']]
                status = "✅" if len(missing) == 0 else "⚠️"
                print(f"  {status} det_{det_model}: {file_count} files" +
                      (f" (missing: {', '.join(missing)})" if missing else ""))
            else:
                print(f"  ❌ det_{det_model}: MISSING")
                report['summary']['issues'].append(f"{dataset}: det_{det_model} missing")

            dataset_report['detection'][det_model] = det_info

        # 2. Check Ground Truth Crops
        print(f"\n[2] GROUND TRUTH CROPS")
        crops_path = exp_path / "crops_gt_crops" / "crops"
        crops_info = {
            'path': str(crops_path),
            'exists': crops_path.exists(),
            'splits': {}
        }

        if crops_path.exists():
            for split in ['train', 'val', 'test']:
                split_path = crops_path / split
                if split_path.exists():
                    # Count images in all class folders
                    total_images = sum(len(list(class_dir.glob("*.png")))
                                      for class_dir in split_path.iterdir()
                                      if class_dir.is_dir())
                    crops_info['splits'][split] = {
                        'exists': True,
                        'image_count': total_images
                    }
                    dataset_report['file_count'] += total_images
                    print(f"  ✅ {split}: {total_images} images")
                else:
                    crops_info['splits'][split] = {'exists': False}
                    print(f"  ❌ {split}: MISSING")
        else:
            print(f"  ❌ crops_gt_crops: MISSING")
            report['summary']['issues'].append(f"{dataset}: Ground truth crops missing")

        dataset_report['crops'] = crops_info

        # 3. Check Classification Models
        print(f"\n[3] CLASSIFICATION MODELS (6 expected)")
        for cls_model in classification_models:
            cls_path = exp_path / f"cls_{cls_model}"
            cls_info = {
                'path': str(cls_path),
                'exists': cls_path.exists(),
                'files': {}
            }

            if cls_path.exists():
                expected_files = {
                    'best.pt': 'Best model weights',
                    'last.pt': 'Last checkpoint',
                    'results.csv': 'Training metrics',
                    'table9_metrics.json': 'Classification metrics',
                    'confusion_matrix.png': 'Confusion matrix',
                    'training_curves.png': 'Training curves'
                }

                for file, desc in expected_files.items():
                    file_path = cls_path / file
                    cls_info['files'][file] = check_file_exists(file_path, desc)
                    if file_path.exists():
                        dataset_report['file_count'] += 1
                        dataset_report['total_size_mb'] += file_path.stat().st_size / (1024**2)

                missing = [f for f, info in cls_info['files'].items() if not info['exists']]
                status = "✅" if len(missing) == 0 else "⚠️"
                print(f"  {status} cls_{cls_model}: " +
                      (f"OK" if not missing else f"missing: {', '.join(missing)}"))
            else:
                print(f"  ❌ cls_{cls_model}: MISSING")
                report['summary']['issues'].append(f"{dataset}: cls_{cls_model} missing")

            dataset_report['classification'][cls_model] = cls_info

        # 4. Check Analysis Folders
        print(f"\n[4] ANALYSIS FOLDERS")
        expected_analysis = (
            [f"analysis_detection_{model}" for model in detection_models] +
            [f"analysis_classification_{model}" for model in classification_models] +
            ['analysis_detection_comparison', 'analysis_dataset_statistics', 'analysis_option_a_summary']
        )

        for analysis_name in expected_analysis:
            analysis_path = exp_path / analysis_name
            exists = analysis_path.exists()
            file_count = len(list(analysis_path.rglob("*"))) if exists else 0

            dataset_report['analysis'][analysis_name] = {
                'exists': exists,
                'file_count': file_count
            }

            status = "✅" if exists else "❌"
            print(f"  {status} {analysis_name}: {'OK' if exists else 'MISSING'}")

            if not exists:
                report['summary']['issues'].append(f"{dataset}: {analysis_name} missing")

        # 5. Check Visualizations
        print(f"\n[5] VISUALIZATIONS")
        viz_path = exp_path / "visualizations"

        if viz_path.exists():
            expected_viz = (
                ['gt_detection', 'gt_classification'] +
                [f"pred_detection_{model}" for model in detection_models] +
                [f"pred_classification_{model}" for model in classification_models]
            )

            for viz_name in expected_viz:
                viz_folder = viz_path / viz_name
                exists = viz_folder.exists()
                image_count = len(list(viz_folder.glob("*.png"))) if exists else 0

                dataset_report['visualizations'][viz_name] = {
                    'exists': exists,
                    'image_count': image_count
                }

                status = "✅" if exists and image_count > 0 else "⚠️" if exists else "❌"
                print(f"  {status} {viz_name}: {image_count} images")

                if not exists or image_count == 0:
                    report['summary']['issues'].append(f"{dataset}: {viz_name} missing or empty")
        else:
            print(f"  ❌ visualizations: MISSING")
            report['summary']['issues'].append(f"{dataset}: visualizations folder missing")

        # 6. Check Summary Files
        print(f"\n[6] SUMMARY FILES")
        summary_files = {
            'master_summary.json': 'Master summary JSON',
            'master_summary.xlsx': 'Master summary Excel',
            'table9_classification_pivot.xlsx': 'Classification pivot table',
            'table9_focal_loss.csv': 'Focal loss metrics',
            'experiment_info.yaml': 'Experiment info'
        }

        for file, desc in summary_files.items():
            file_path = exp_path / file
            file_info = check_file_exists(file_path, desc)
            dataset_report['summary_files'][file] = file_info

            status = "✅" if file_info['exists'] else "❌"
            print(f"  {status} {file}")

            if not file_info['exists']:
                report['summary']['issues'].append(f"{dataset}: {file} missing")

        # Dataset summary
        print(f"\n{'='*80}")
        print(f"DATASET SUMMARY: {dataset.upper()}")
        print(f"  Total files: {dataset_report['file_count']}")
        print(f"  Total size: {dataset_report['total_size_mb']:.2f} MB")
        print(f"{'='*80}")

        report['datasets'][dataset] = dataset_report
        report['summary']['total_files'] += dataset_report['file_count']
        report['summary']['total_size_mb'] += dataset_report['total_size_mb']

    # Check Consolidated Analysis
    print(f"\n{'='*80}")
    print(f"CONSOLIDATED ANALYSIS (Cross-Dataset)")
    print(f"{'='*80}")

    cons_path = base_path / "consolidated_analysis" / "cross_dataset_comparison"

    if cons_path.exists():
        expected_consolidated = {
            'classification_performance_all_datasets.xlsx': 'Classification comparison',
            'detection_performance_all_datasets.xlsx': 'Detection comparison',
            'dataset_statistics_all.csv': 'Dataset statistics',
            'comprehensive_summary.json': 'Comprehensive summary',
            'README.md': 'Consolidated README'
        }

        for file, desc in expected_consolidated.items():
            file_path = cons_path / file
            file_info = check_file_exists(file_path, desc)
            report['consolidated_analysis'][file] = file_info

            status = "✅" if file_info['exists'] else "❌"
            print(f"  {status} {file}")

            if not file_info['exists']:
                report['summary']['issues'].append(f"Consolidated: {file} missing")
    else:
        print(f"  ❌ consolidated_analysis: MISSING")
        report['summary']['issues'].append("Consolidated analysis folder missing")

    # Final Summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETENESS SUMMARY")
    print(f"{'='*80}")
    print(f"Total files across all datasets: {report['summary']['total_files']}")
    print(f"Total size: {report['summary']['total_size_mb']:.2f} MB ({report['summary']['total_size_mb']/1024:.2f} GB)")
    print(f"Datasets processed: {len(report['datasets'])}/4")
    print(f"Issues found: {len(report['summary']['issues'])}")

    if report['summary']['issues']:
        print(f"\n⚠️ ISSUES FOUND:")
        for issue in report['summary']['issues']:
            print(f"  - {issue}")
    else:
        print(f"\n✅ ALL EXPECTED OUTPUTS PRESENT!")

    # Save detailed report
    report_path = base_path / "completeness_verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📊 Detailed report saved: {report_path}")

    return report

if __name__ == "__main__":
    report = analyze_experiment()
