#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate COMPREHENSIVE Consolidated Multi-Dataset Analysis
Includes: Dataset statistics, Detection performance, Classification performance, Table 9 comparisons
"""

import sys
import json
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')

def load_dataset_statistics(experiment_path):
    """Load dataset statistics (augmentation data)"""
    stats_file = experiment_path / "analysis_dataset_statistics" / "dataset_statistics_summary.csv"
    if stats_file.exists():
        df = pd.read_csv(stats_file)
        return df.to_dict('records')
    return None

def load_detection_performance(experiment_path):
    """Load detection models performance"""
    det_file = experiment_path / "analysis_detection_comparison" / "detection_models_summary.json"
    if det_file.exists():
        with open(det_file, 'r') as f:
            return json.load(f)
    return None

def load_table9_data(experiment_path):
    """Load Table 9 classification pivot data (CE, Focal, Class-Balanced)"""
    table9_ce = experiment_path / "table9_cross_entropy.csv"
    table9_focal = experiment_path / "table9_focal_loss.csv"
    table9_cb = experiment_path / "table9_class_balanced.csv"  # NEW

    data = {}
    if table9_ce.exists():
        df_ce = pd.read_csv(table9_ce)
        data['cross_entropy'] = df_ce.to_dict('records')

    if table9_focal.exists():
        df_focal = pd.read_csv(table9_focal)
        data['focal_loss'] = df_focal.to_dict('records')

    if table9_cb.exists():
        df_cb = pd.read_csv(table9_cb)
        data['class_balanced'] = df_cb.to_dict('records')

    return data if data else None

def generate_comprehensive_consolidated_analysis(results_dir: str):
    """Generate comprehensive consolidated analysis"""

    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return False

    experiments_dir = results_path / "experiments"
    if not experiments_dir.exists():
        print(f"[ERROR] Not a multi-dataset experiment.")
        return False

    dataset_folders = sorted([f for f in experiments_dir.iterdir()
                             if f.is_dir() and f.name.startswith("experiment_")])

    if len(dataset_folders) < 2:
        print(f"[ERROR] Need at least 2 datasets. Found: {len(dataset_folders)}")
        return False

    print(f"\n{'='*80}")
    print(f"[COMPREHENSIVE] GENERATING MULTI-DATASET CONSOLIDATED ANALYSIS")
    print(f"{'='*80}")
    print(f"[FOLDER] Results: {results_path}")
    print(f"[DATASETS] Found {len(dataset_folders)} datasets")

    # Collect comprehensive data
    consolidated_data = {
        "parent_experiment": results_path.name,
        "total_datasets": len(dataset_folders),
        "analysis_timestamp": datetime.now().isoformat(),
        "datasets": {}
    }

    # Dataset statistics summary (augmentation data)
    all_dataset_stats = []

    # Detection performance summary
    all_detection_performance = {}

    # Classification performance summary (Table 9)
    all_classification_performance = {}

    print(f"\n[COLLECT] Gathering data from all datasets...")

    for folder in dataset_folders:
        dataset_name = folder.name.replace("experiment_", "")
        print(f"   [LOAD] {dataset_name}...")

        # 1. Dataset Statistics (Augmentation)
        stats = load_dataset_statistics(folder)
        if stats:
            all_dataset_stats.extend(stats)

        # 2. Detection Performance
        det_perf = load_detection_performance(folder)
        if det_perf:
            all_detection_performance[dataset_name] = det_perf.get('models_performance', {})

        # 3. Classification Performance (Table 9)
        table9 = load_table9_data(folder)
        if table9:
            all_classification_performance[dataset_name] = table9

        # Store in consolidated structure
        consolidated_data["datasets"][dataset_name] = {
            "path": str(folder),
            "has_detection_analysis": det_perf is not None,
            "has_classification_analysis": table9 is not None,
            "has_dataset_statistics": stats is not None
        }

    # Create consolidated analysis folder
    consolidated_path = results_path / "consolidated_analysis" / "cross_dataset_comparison"
    consolidated_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[CREATE] Consolidated folder: {consolidated_path}")

    # ========================================
    # 1. SAVE DATASET STATISTICS (AUGMENTATION)
    # ========================================
    if all_dataset_stats:
        stats_df = pd.DataFrame(all_dataset_stats)
        stats_csv = consolidated_path / "dataset_statistics_all.csv"
        stats_df.to_csv(stats_csv, index=False)
        print(f"[SAVE] Dataset statistics: {stats_csv.name}")

        # Add to consolidated data
        consolidated_data["dataset_statistics"] = all_dataset_stats

    # ========================================
    # 2. SAVE DETECTION PERFORMANCE COMPARISON
    # ========================================
    if all_detection_performance:
        # Create comparison table
        detection_comparison = []
        for dataset, models in all_detection_performance.items():
            for model_name, metrics in models.items():
                row = {
                    "Dataset": dataset,
                    "Model": model_name.upper(),
                    "Epochs": metrics.get("epochs_trained", 0),
                    "mAP@50": round(metrics.get("mAP50", 0), 4),
                    "mAP@50-95": round(metrics.get("mAP50_95", 0), 4),
                    "Precision": round(metrics.get("precision", 0), 4),
                    "Recall": round(metrics.get("recall", 0), 4)
                }
                detection_comparison.append(row)

        det_df = pd.DataFrame(detection_comparison)

        # Save CSV
        det_csv = consolidated_path / "detection_performance_all_datasets.csv"
        det_df.to_csv(det_csv, index=False)
        print(f"[SAVE] Detection comparison: {det_csv.name}")

        # Save Excel with formatting
        det_xlsx = consolidated_path / "detection_performance_all_datasets.xlsx"
        det_df.to_excel(det_xlsx, index=False, sheet_name="All Datasets")
        print(f"[SAVE] Detection Excel: {det_xlsx.name}")

        # Add to consolidated data
        consolidated_data["detection_performance"] = all_detection_performance

    # ========================================
    # 3. SAVE CLASSIFICATION PERFORMANCE (TABLE 9)
    # ========================================
    if all_classification_performance:
        # Cross-Entropy, Focal Loss, and Class-Balanced Comparison
        ce_comparison = []
        focal_comparison = []
        cb_comparison = []  # NEW

        for dataset, table9_data in all_classification_performance.items():
            if 'cross_entropy' in table9_data:
                for row in table9_data['cross_entropy']:
                    row_copy = row.copy()
                    row_copy['Dataset'] = dataset
                    ce_comparison.append(row_copy)

            if 'focal_loss' in table9_data:
                for row in table9_data['focal_loss']:
                    row_copy = row.copy()
                    row_copy['Dataset'] = dataset
                    focal_comparison.append(row_copy)

            if 'class_balanced' in table9_data:
                for row in table9_data['class_balanced']:
                    row_copy = row.copy()
                    row_copy['Dataset'] = dataset
                    cb_comparison.append(row_copy)

        # Save Cross-Entropy comparison
        if ce_comparison:
            ce_df = pd.DataFrame(ce_comparison)
            ce_csv = consolidated_path / "classification_cross_entropy_all_datasets.csv"
            ce_df.to_csv(ce_csv, index=False)
            print(f"[SAVE] Classification (CE): {ce_csv.name}")

        # Save Focal Loss comparison
        if focal_comparison:
            focal_df = pd.DataFrame(focal_comparison)
            focal_csv = consolidated_path / "classification_focal_loss_all_datasets.csv"
            focal_df.to_csv(focal_csv, index=False)
            print(f"[SAVE] Classification (Focal): {focal_csv.name}")

        # Save Class-Balanced comparison
        if cb_comparison:
            cb_df = pd.DataFrame(cb_comparison)
            cb_csv = consolidated_path / "classification_class_balanced_all_datasets.csv"
            cb_df.to_csv(cb_csv, index=False)
            print(f"[SAVE] Classification (Class-Balanced): {cb_csv.name}")

        # Save combined Excel (3 sheets now)
        if ce_comparison or focal_comparison or cb_comparison:
            classification_xlsx = consolidated_path / "classification_performance_all_datasets.xlsx"
            with pd.ExcelWriter(classification_xlsx) as writer:
                if ce_comparison:
                    ce_df.to_excel(writer, sheet_name="Cross-Entropy", index=False)
                if focal_comparison:
                    focal_df.to_excel(writer, sheet_name="Focal Loss", index=False)
                if cb_comparison:
                    cb_df.to_excel(writer, sheet_name="Class-Balanced", index=False)
            print(f"[SAVE] Classification Excel: {classification_xlsx.name}")

        # Add to consolidated data
        consolidated_data["classification_performance"] = all_classification_performance

    # ========================================
    # 4. SAVE COMPREHENSIVE JSON
    # ========================================
    consolidated_json = consolidated_path / "comprehensive_summary.json"
    with open(consolidated_json, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=2)
    print(f"[SAVE] Comprehensive JSON: {consolidated_json.name}")

    # ========================================
    # 5. CREATE COMPREHENSIVE README
    # ========================================
    readme_content = f"""# Comprehensive Multi-Dataset Consolidated Analysis

## Experiment Overview
- **Parent Experiment**: {results_path.name}
- **Total Datasets**: {len(dataset_folders)}
- **Datasets Analyzed**: {', '.join([f.name.replace('experiment_', '') for f in dataset_folders])}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Dataset Statistics (Augmentation Effects)

"""

    if all_dataset_stats:
        readme_content += "| Dataset | Original Train | Original Val | Original Test | Detection Aug | Classification Aug | Det Multiplier | Cls Multiplier |\n"
        readme_content += "|---------|----------------|--------------|---------------|---------------|-------------------|----------------|----------------|\n"

        for stat in all_dataset_stats:
            readme_content += f"| {stat['Dataset']} | {stat['Original_Train']} | {stat['Original_Val']} | {stat['Original_Test']} | {stat['Detection_Aug_Train']} | {stat['Classification_Aug_Train']} | {stat['Detection_Multiplier']} | {stat['Classification_Multiplier']} |\n"

    readme_content += "\n---\n\n## 🎯 Detection Model Performance\n\n"

    if all_detection_performance:
        for dataset, models in all_detection_performance.items():
            readme_content += f"\n### {dataset.upper()}\n\n"
            readme_content += "| Model | mAP@50 | mAP@50-95 | Precision | Recall |\n"
            readme_content += "|-------|--------|-----------|-----------|--------|\n"

            for model_name, metrics in models.items():
                readme_content += f"| {model_name.upper()} | {metrics.get('mAP50', 0):.4f} | {metrics.get('mAP50_95', 0):.4f} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} |\n"

    readme_content += "\n---\n\n## 🧬 Classification Model Performance (Table 9 Summary)\n\n"

    if all_classification_performance:
        for dataset, table9_data in all_classification_performance.items():
            readme_content += f"\n### {dataset.upper()}\n\n"

            if 'cross_entropy' in table9_data:
                overall_ce = [row for row in table9_data['cross_entropy'] if row.get('Class') == 'Overall' and row.get('Metric') == 'accuracy']
                if overall_ce:
                    readme_content += "**Cross-Entropy:**\n"
                    row = overall_ce[0]
                    for key, val in row.items():
                        if key not in ['Class', 'Metric', 'Dataset']:
                            readme_content += f"- {key}: {val}\n"

            if 'focal_loss' in table9_data:
                overall_focal = [row for row in table9_data['focal_loss'] if row.get('Class') == 'Overall' and row.get('Metric') == 'accuracy']
                if overall_focal:
                    readme_content += "\n**Focal Loss:**\n"
                    row = overall_focal[0]
                    for key, val in row.items():
                        if key not in ['Class', 'Metric', 'Dataset']:
                            readme_content += f"- {key}: {val}\n"

            if 'class_balanced' in table9_data:
                overall_cb = [row for row in table9_data['class_balanced'] if row.get('Class') == 'Overall' and row.get('Metric') == 'accuracy']
                if overall_cb:
                    readme_content += "\n**Class-Balanced:**\n"
                    row = overall_cb[0]
                    for key, val in row.items():
                        if key not in ['Class', 'Metric', 'Dataset']:
                            readme_content += f"- {key}: {val}\n"

    readme_content += """
---

## 📁 Files Generated

### Dataset Statistics:
- `dataset_statistics_all.csv` - Augmentation effects across all datasets

### Detection Performance:
- `detection_performance_all_datasets.csv` - Detection comparison (CSV)
- `detection_performance_all_datasets.xlsx` - Detection comparison (Excel)

### Classification Performance:
- `classification_cross_entropy_all_datasets.csv` - Cross-Entropy results
- `classification_focal_loss_all_datasets.csv` - Focal Loss results
- `classification_class_balanced_all_datasets.csv` - Class-Balanced results
- `classification_performance_all_datasets.xlsx` - Combined Excel (3 sheets)

### Summary:
- `comprehensive_summary.json` - Complete data in JSON format
- `README.md` - This overview file

---

## 📈 How to Use

1. **Dataset Comparison**: Check `dataset_statistics_all.csv` for augmentation effects
2. **Detection Models**: Review `detection_performance_all_datasets.xlsx` for YOLO comparisons
3. **Classification Models**: Open `classification_performance_all_datasets.xlsx` for detailed analysis (3 loss functions)
4. **Raw Data**: Use `comprehensive_summary.json` for programmatic access

---

*Generated by Comprehensive Consolidated Analysis Script*
*Timestamp: {datetime.now().isoformat()}*
"""

    readme_file = consolidated_path / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"[SAVE] README: {readme_file.name}")

    print(f"\n{'='*80}")
    print(f"[SUCCESS] COMPREHENSIVE CONSOLIDATED ANALYSIS COMPLETED!")
    print(f"{'='*80}")
    print(f"[LOCATION] {consolidated_path}")
    print(f"\n[FILES] Generated:")
    print(f"   📊 Dataset Statistics:")
    print(f"      - dataset_statistics_all.csv")
    print(f"   🎯 Detection Performance:")
    print(f"      - detection_performance_all_datasets.csv")
    print(f"      - detection_performance_all_datasets.xlsx")
    print(f"   🧬 Classification Performance (3 Loss Functions):")
    print(f"      - classification_cross_entropy_all_datasets.csv")
    print(f"      - classification_focal_loss_all_datasets.csv")
    print(f"      - classification_class_balanced_all_datasets.csv")
    print(f"      - classification_performance_all_datasets.xlsx (3 sheets)")
    print(f"   📝 Summary:")
    print(f"      - comprehensive_summary.json")
    print(f"      - README.md")

    return True

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive consolidated multi-dataset analysis")
    parser.add_argument(
        "results_dir",
        help="Path to multi-dataset experiment results (e.g., results/optA_20251001_095625)"
    )

    args = parser.parse_args()
    success = generate_comprehensive_consolidated_analysis(args.results_dir)

    if not success:
        exit(1)

if __name__ == "__main__":
    main()
