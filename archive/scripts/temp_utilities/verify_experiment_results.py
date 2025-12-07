import pandas as pd
import json
import os

base = r"C:\Users\MyPC PRO\Documents\hello_world\results\optA_20251016_200330\experiments"

print("="*100)
print("VERIFICATION: PAPER vs ACTUAL EXPERIMENT RESULTS")
print("="*100)

# ============================================================================
# DETECTION RESULTS VERIFICATION
# ============================================================================
print("\n" + "="*100)
print("DETECTION PERFORMANCE VERIFICATION")
print("="*100)

# IML Lifecycle - YOLO11
print("\n[1] IML LIFECYCLE - YOLO11")
print("-" * 50)
iml_yolo11 = pd.read_csv(os.path.join(base, "experiment_iml_lifecycle/det_yolo11/results.csv"))
iml_yolo11 = iml_yolo11.tail(10)  # Last 10 epochs
print("Last 10 epochs:")
print(iml_yolo11[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].to_string(index=False))

# Find best epoch
best_epoch_iml = iml_yolo11.loc[iml_yolo11['metrics/mAP50(B)'].idxmax()]
print(f"\n✅ BEST EPOCH: {int(best_epoch_iml['epoch'])}")
print(f"   mAP@50: {best_epoch_iml['metrics/mAP50(B)']:.4f} ({best_epoch_iml['metrics/mAP50(B)']*100:.2f}%)")
print(f"   mAP@50-95: {best_epoch_iml['metrics/mAP50-95(B)']:.4f} ({best_epoch_iml['metrics/mAP50-95(B)']*100:.2f}%)")
print(f"\n📄 PAPER CLAIMS: 94.99% mAP@50")
print(f"✅ ACTUAL: {best_epoch_iml['metrics/mAP50(B)']*100:.2f}%")
print(f"{'✅ MATCH!' if abs(best_epoch_iml['metrics/mAP50(B)']*100 - 94.99) < 0.1 else '❌ MISMATCH!'}")

# MD_2019 - YOLO11
print("\n[2] MD_2019 STAGES - YOLO11")
print("-" * 50)
md_yolo11 = pd.read_csv(os.path.join(base, "experiment_md_2019_stages/det_yolo11/results.csv"))
md_yolo11 = md_yolo11.tail(10)
print("Last 10 epochs:")
print(md_yolo11[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].to_string(index=False))

best_epoch_md = md_yolo11.loc[md_yolo11['metrics/mAP50(B)'].idxmax()]
print(f"\n✅ BEST EPOCH: {int(best_epoch_md['epoch'])}")
print(f"   mAP@50: {best_epoch_md['metrics/mAP50(B)']:.4f} ({best_epoch_md['metrics/mAP50(B)']*100:.2f}%)")
print(f"\n📄 PAPER CLAIMS: 72.91% mAP@50")
print(f"✅ ACTUAL: {best_epoch_md['metrics/mAP50(B)']*100:.2f}%")
print(f"{'✅ MATCH!' if abs(best_epoch_md['metrics/mAP50(B)']*100 - 72.91) < 0.1 else '❌ MISMATCH!'}")

# MP-IDB Stages - YOLO12
print("\n[3] MP-IDB STAGES - YOLO12")
print("-" * 50)
if os.path.exists(os.path.join(base, "experiment_mp_idb_stages/det_yolo12/results.csv")):
    stages_yolo12 = pd.read_csv(os.path.join(base, "experiment_mp_idb_stages/det_yolo12/results.csv"))
    stages_yolo12 = stages_yolo12.tail(10)
    print("Last 10 epochs:")
    print(stages_yolo12[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']].to_string(index=False))

    best_epoch_stages = stages_yolo12.loc[stages_yolo12['metrics/mAP50(B)'].idxmax()]
    print(f"\n✅ BEST EPOCH: {int(best_epoch_stages['epoch'])}")
    print(f"   mAP@50: {best_epoch_stages['metrics/mAP50(B)']:.4f} ({best_epoch_stages['metrics/mAP50(B)']*100:.2f}%)")
    print(f"\n📄 PAPER CLAIMS: 96.27% mAP@50")
    print(f"✅ ACTUAL: {best_epoch_stages['metrics/mAP50(B)']*100:.2f}%")
    print(f"{'✅ MATCH!' if abs(best_epoch_stages['metrics/mAP50(B)']*100 - 96.27) < 0.1 else '❌ MISMATCH!'}")

# ============================================================================
# CLASSIFICATION RESULTS VERIFICATION
# ============================================================================
print("\n\n" + "="*100)
print("CLASSIFICATION PERFORMANCE VERIFICATION")
print("="*100)

# IML Lifecycle - EfficientNet-B1
print("\n[4] IML LIFECYCLE - EFFICIENTNET-B1")
print("-" * 50)
iml_table9 = pd.read_csv(os.path.join(base, "experiment_iml_lifecycle/table9_focal_loss.csv"))
print(iml_table9.to_string(index=False))

effb1_iml = iml_table9[iml_table9['Model'].str.contains('EfficientNet-B1', case=False, na=False)]
if not effb1_iml.empty:
    accuracy = effb1_iml['Overall Accuracy'].values[0]
    print(f"\n📄 PAPER CLAIMS: 91.51% accuracy")
    print(f"✅ ACTUAL: {accuracy*100:.2f}%")
    print(f"{'✅ MATCH!' if abs(accuracy*100 - 91.51) < 0.1 else '❌ MISMATCH!'}")

# MP-IDB Species - EfficientNet-B1
print("\n[5] MP-IDB SPECIES - EFFICIENTNET-B1")
print("-" * 50)
species_table9 = pd.read_csv(os.path.join(base, "experiment_mp_idb_species/table9_focal_loss.csv"))
print(species_table9.to_string(index=False))

effb1_species = species_table9[species_table9['Model'].str.contains('EfficientNet-B1', case=False, na=False)]
if not effb1_species.empty:
    accuracy = effb1_species['Overall Accuracy'].values[0]
    print(f"\n📄 PAPER CLAIMS: 98.28% accuracy")
    print(f"✅ ACTUAL: {accuracy*100:.2f}%")
    print(f"{'✅ MATCH!' if abs(accuracy*100 - 98.28) < 0.1 else '❌ MISMATCH!'}")

# MP-IDB Stages - ResNet50
print("\n[6] MP-IDB STAGES - RESNET50")
print("-" * 50)
stages_table9 = pd.read_csv(os.path.join(base, "experiment_mp_idb_stages/table9_focal_loss.csv"))
print(stages_table9.to_string(index=False))

res50_stages = stages_table9[stages_table9['Model'].str.contains('ResNet50', case=False, na=False)]
# Exclude ResNet101
res50_stages = res50_stages[~res50_stages['Model'].str.contains('101', na=False)]
if not res50_stages.empty:
    accuracy = res50_stages['Overall Accuracy'].values[0]
    print(f"\n📄 PAPER CLAIMS: 96.13% accuracy")
    print(f"✅ ACTUAL: {accuracy*100:.2f}%")
    print(f"{'✅ MATCH!' if abs(accuracy*100 - 96.13) < 0.1 else '❌ MISMATCH!'}")

# MD_2019 - EfficientNet-B0
print("\n[7] MD_2019 STAGES - EFFICIENTNET-B0")
print("-" * 50)
md_table9 = pd.read_csv(os.path.join(base, "experiment_md_2019_stages/table9_focal_loss.csv"))
print(md_table9.to_string(index=False))

effb0_md = md_table9[md_table9['Model'].str.contains('EfficientNet-B0', case=False, na=False)]
if not effb0_md.empty:
    accuracy = effb0_md['Overall Accuracy'].values[0]
    print(f"\n📄 PAPER CLAIMS: 86.45% accuracy")
    print(f"✅ ACTUAL: {accuracy*100:.2f}%")
    print(f"{'✅ MATCH!' if abs(accuracy*100 - 86.45) < 0.1 else '❌ MISMATCH!'}")

print("\n" + "="*100)
print("✅ VERIFICATION COMPLETE!")
print("="*100)
