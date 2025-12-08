import json
import pandas as pd

# Load both comprehensive summaries
baseline = json.load(open('results/optA_20251016_200330/consolidated_analysis/cross_dataset_comparison/comprehensive_summary.json'))
current = json.load(open('results/optA_20251207_233941/consolidated_analysis/cross_dataset_comparison/comprehensive_summary.json'))

print("="*80)
print("PERBANDINGAN AKURASI: BASELINE (Oct 16) vs CURRENT (Dec 7)")
print("="*80)
print()

datasets = ['iml_lifecycle', 'md_2019_stages', 'mp_idb_species', 'mp_idb_stages']

# Detection Comparison
print("🎯 DETECTION PERFORMANCE (mAP@50)")
print("-"*80)
for dataset in datasets:
    print(f"\n{dataset.upper().replace('_', ' ')}:")
    for model in ['yolo10', 'yolo11', 'yolo12']:
        base_val = baseline['detection_performance'][dataset][model]['mAP50']
        curr_val = current['detection_performance'][dataset][model]['mAP50']
        diff = (curr_val - base_val) * 100

        if diff > 0:
            status = f"✅ +{diff:.2f}%"
        elif diff < -0.5:
            status = f"⚠️ {diff:.2f}%"
        else:
            status = f"⚪ {diff:.2f}%"

        print(f"  {model.upper():8s}: {base_val*100:6.2f}% → {curr_val*100:6.2f}% ({status})")

# Classification Comparison
print("\n" + "="*80)
print("🎓 CLASSIFICATION PERFORMANCE (Accuracy)")
print("-"*80)

models = ['densenet121', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'resnet101', 'resnet50']

for dataset in datasets:
    print(f"\n{dataset.upper().replace('_', ' ')}:")

    # Get accuracy from focal_loss array
    base_focal = baseline['classification_performance'][dataset]['focal_loss']
    curr_focal = current['classification_performance'][dataset]['focal_loss']

    # Find accuracy row
    base_acc_row = [r for r in base_focal if r['Class'] == 'Overall' and r['Metric'] == 'accuracy'][0]
    curr_acc_row = [r for r in curr_focal if r['Class'] == 'Overall' and r['Metric'] == 'accuracy'][0]

    for model in models:
        base_val = base_acc_row[model]
        curr_val = curr_acc_row[model]
        diff = (curr_val - base_val) * 100

        if diff > 0.5:
            status = f"✅ +{diff:.2f}%"
        elif diff < -0.5:
            status = f"⚠️ {diff:.2f}%"
        else:
            status = f"⚪ {diff:.2f}%"

        print(f"  {model:18s}: {base_val*100:6.2f}% → {curr_val*100:6.2f}% ({status})")

# Balanced Accuracy Comparison (important for medical AI)
print("\n" + "="*80)
print("⚖️  BALANCED ACCURACY (Critical for Medical AI)")
print("-"*80)

for dataset in datasets:
    print(f"\n{dataset.upper().replace('_', ' ')}:")

    base_focal = baseline['classification_performance'][dataset]['focal_loss']
    curr_focal = current['classification_performance'][dataset]['focal_loss']

    base_bal_row = [r for r in base_focal if r['Class'] == 'Overall' and r['Metric'] == 'balanced_accuracy'][0]
    curr_bal_row = [r for r in curr_focal if r['Class'] == 'Overall' and r['Metric'] == 'balanced_accuracy'][0]

    for model in models:
        base_val = base_bal_row[model]
        curr_val = curr_bal_row[model]
        diff = (curr_val - base_val) * 100

        if diff > 0.5:
            status = f"✅ +{diff:.2f}%"
        elif diff < -1.0:
            status = f"⚠️ {diff:.2f}%"
        else:
            status = f"⚪ {diff:.2f}%"

        print(f"  {model:18s}: {base_val*100:6.2f}% → {curr_val*100:6.2f}% ({status})")

# Summary Statistics
print("\n" + "="*80)
print("📊 SUMMARY STATISTICS")
print("="*80)

# Count improvements, stable, and decreases
det_improvements = 0
det_decreases = 0
det_stable = 0

for dataset in datasets:
    for model in ['yolo10', 'yolo11', 'yolo12']:
        base_val = baseline['detection_performance'][dataset][model]['mAP50']
        curr_val = current['detection_performance'][dataset][model]['mAP50']
        diff = curr_val - base_val

        if diff > 0.005:
            det_improvements += 1
        elif diff < -0.005:
            det_decreases += 1
        else:
            det_stable += 1

cls_improvements = 0
cls_decreases = 0
cls_stable = 0

for dataset in datasets:
    base_focal = baseline['classification_performance'][dataset]['focal_loss']
    curr_focal = current['classification_performance'][dataset]['focal_loss']

    base_acc_row = [r for r in base_focal if r['Class'] == 'Overall' and r['Metric'] == 'accuracy'][0]
    curr_acc_row = [r for r in curr_focal if r['Class'] == 'Overall' and r['Metric'] == 'accuracy'][0]

    for model in models:
        base_val = base_acc_row[model]
        curr_val = curr_acc_row[model]
        diff = curr_val - base_val

        if diff > 0.005:
            cls_improvements += 1
        elif diff < -0.005:
            cls_decreases += 1
        else:
            cls_stable += 1

print(f"\nDETECTION (12 total experiments):")
print(f"  ✅ Improved:  {det_improvements:2d} ({det_improvements/12*100:.1f}%)")
print(f"  ⚪ Stable:    {det_stable:2d} ({det_stable/12*100:.1f}%)")
print(f"  ⚠️  Decreased: {det_decreases:2d} ({det_decreases/12*100:.1f}%)")

print(f"\nCLASSIFICATION (24 total experiments):")
print(f"  ✅ Improved:  {cls_improvements:2d} ({cls_improvements/24*100:.1f}%)")
print(f"  ⚪ Stable:    {cls_stable:2d} ({cls_stable/24*100:.1f}%)")
print(f"  ⚠️  Decreased: {cls_decreases:2d} ({cls_decreases/24*100:.1f}%)")

print("\n" + "="*80)
