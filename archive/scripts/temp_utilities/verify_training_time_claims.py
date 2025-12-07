"""
Verify training time claims in the paper
"""
import pandas as pd
import re
from pathlib import Path

results_dir = Path("results/optA_20251016_200330/experiments")

# Datasets
datasets = ["iml_lifecycle", "mp_idb_species", "mp_idb_stages", "md_2019_stages"]

# Detection models
det_models = ["yolo10", "yolo11", "yolo12"]

# Classification models
cls_models = ["densenet121", "efficientnet_b0", "efficientnet_b1",
              "efficientnet_b2", "resnet50", "resnet101"]

print("="*80)
print("TRAINING TIME VERIFICATION")
print("="*80)

# ============================================================================
# 1. DETECTION TRAINING TIME
# ============================================================================
print("\n1. DETECTION TRAINING TIME")
print("-" * 80)

total_det_time_hours = 0
det_times = {}

for dataset in datasets:
    det_times[dataset] = {}
    for model in det_models:
        csv_path = results_dir / f"experiment_{dataset}" / f"det_{model}" / "results.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Last epoch's cumulative time (in seconds)
            final_time_sec = df['time'].iloc[-1]
            final_time_hours = final_time_sec / 3600

            det_times[dataset][model] = final_time_hours
            total_det_time_hours += final_time_hours

            print(f"  {dataset:20s} {model:8s}: {final_time_hours:6.2f} hours ({final_time_sec/60:6.1f} min)")

print(f"\n  TOTAL DETECTION TIME: {total_det_time_hours:.2f} hours")

# ============================================================================
# 2. CLASSIFICATION TRAINING TIME (SHARED APPROACH)
# ============================================================================
print("\n2. CLASSIFICATION TRAINING TIME (Shared Approach)")
print("-" * 80)

total_cls_time_hours = 0
cls_times = {}

for dataset in datasets:
    cls_times[dataset] = {}
    for model in cls_models:
        results_path = results_dir / f"experiment_{dataset}" / f"cls_{model}_focal" / "results.txt"

        if results_path.exists():
            with open(results_path, 'r') as f:
                content = f.read()

            # Extract training time (format: "Training Time: X.X min")
            match = re.search(r'Training Time:\s+([\d.]+)\s+min', content)
            if match:
                time_min = float(match.group(1))
                time_hours = time_min / 60

                cls_times[dataset][model] = time_hours
                total_cls_time_hours += time_hours

                print(f"  {dataset:20s} {model:15s}: {time_hours:6.2f} hours ({time_min:6.1f} min)")

print(f"\n  TOTAL CLASSIFICATION TIME: {total_cls_time_hours:.2f} hours")

# ============================================================================
# 3. TOTAL SHARED APPROACH TIME
# ============================================================================
print("\n3. TOTAL SHARED APPROACH TIME")
print("-" * 80)

total_shared_time = total_det_time_hours + total_cls_time_hours
print(f"  Detection:      {total_det_time_hours:6.2f} hours")
print(f"  Classification: {total_cls_time_hours:6.2f} hours")
print(f"  TOTAL:          {total_shared_time:6.2f} hours")

# ============================================================================
# 4. TRADITIONAL APPROACH ESTIMATE
# ============================================================================
print("\n4. TRADITIONAL APPROACH ESTIMATE (Detection-Specific Models)")
print("-" * 80)
print("Traditional approach trains:")
print("  - 3 detection models × 4 datasets = 12 detection runs (same as shared)")
print("  - 6 classification models × 3 detection models × 4 datasets = 72 cls runs")
print("")
print("Shared approach trains:")
print("  - 3 detection models × 4 datasets = 12 detection runs")
print("  - 6 classification models × 4 datasets = 24 cls runs")
print("")

# Calculate average classification time per model per dataset
avg_cls_time_per_run = total_cls_time_hours / 24  # 24 classification runs in shared

traditional_det_time = total_det_time_hours  # Same detection time
traditional_cls_time = avg_cls_time_per_run * 72  # 72 cls runs in traditional
traditional_total = traditional_det_time + traditional_cls_time

print(f"Average classification time per run: {avg_cls_time_per_run:.2f} hours")
print(f"\nTraditional Approach Total:")
print(f"  Detection:      {traditional_det_time:6.2f} hours (same as shared)")
print(f"  Classification: {traditional_cls_time:6.2f} hours (72 runs × {avg_cls_time_per_run:.2f}h)")
print(f"  TOTAL:          {traditional_total:6.2f} hours")

# ============================================================================
# 5. COMPARISON WITH PAPER CLAIMS
# ============================================================================
print("\n5. COMPARISON WITH PAPER CLAIMS")
print("="*80)

paper_shared_claim = 18  # hours
paper_traditional_claim = 54  # hours

print(f"\nPAPER CLAIMS:")
print(f"  Shared approach:      {paper_shared_claim:6.1f} hours")
print(f"  Traditional approach: {paper_traditional_claim:6.1f} hours")
print(f"  Reduction:            {(1 - paper_shared_claim/paper_traditional_claim)*100:.1f}%")

print(f"\nACTUAL DATA:")
print(f"  Shared approach:      {total_shared_time:6.1f} hours")
print(f"  Traditional approach: {traditional_total:6.1f} hours (estimated)")
print(f"  Reduction:            {(1 - total_shared_time/traditional_total)*100:.1f}%")

print(f"\nDISCREPANCY:")
print(f"  Shared claim vs actual:      {paper_shared_claim:6.1f}h vs {total_shared_time:6.1f}h = {abs(paper_shared_claim - total_shared_time):5.1f}h off")
print(f"  Traditional claim vs actual: {paper_traditional_claim:6.1f}h vs {traditional_total:6.1f}h = {abs(paper_traditional_claim - traditional_total):5.1f}h off")

# Determine if claims are acceptable
shared_error_pct = abs(paper_shared_claim - total_shared_time) / total_shared_time * 100
traditional_error_pct = abs(paper_traditional_claim - traditional_total) / traditional_total * 100

print(f"\nERROR PERCENTAGE:")
print(f"  Shared:      {shared_error_pct:5.1f}%")
print(f"  Traditional: {traditional_error_pct:5.1f}%")

if shared_error_pct > 20 or traditional_error_pct > 20:
    print(f"\n⚠️  WARNING: Error > 20% - claims may need correction!")
else:
    print(f"\n✅ OK: Claims within 20% margin")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n1. Shared Approach Total Time: {total_shared_time:.1f} hours")
print(f"   Paper claims: {paper_shared_claim} hours")
print(f"   Status: {'❌ NEEDS CORRECTION' if shared_error_pct > 20 else '✅ ACCEPTABLE'}")

print(f"\n2. Traditional Approach (Estimated): {traditional_total:.1f} hours")
print(f"   Paper claims: {paper_traditional_claim} hours")
print(f"   Status: {'❌ NEEDS CORRECTION' if traditional_error_pct > 20 else '✅ ACCEPTABLE'}")

print(f"\n3. Training Time Reduction: {(1 - total_shared_time/traditional_total)*100:.1f}%")
print(f"   Paper claims: 67% reduction")

print("\n" + "="*80)
