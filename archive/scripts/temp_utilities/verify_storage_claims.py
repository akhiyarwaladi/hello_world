"""
Verify storage size claims in the paper
"""
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
print("STORAGE SIZE VERIFICATION")
print("="*80)

# ============================================================================
# 1. CLASSIFICATION MODEL WEIGHTS (SHARED APPROACH)
# ============================================================================
print("\n1. CLASSIFICATION MODEL WEIGHTS (Shared Approach)")
print("-" * 80)

total_cls_weights_mb = 0

for dataset in datasets:
    print(f"\n{dataset.upper()}:")
    dataset_total = 0

    for model in cls_models:
        model_dir = results_dir / f"experiment_{dataset}" / f"cls_{model}_focal"
        best_pt = model_dir / "best.pt"

        if best_pt.exists():
            size_mb = best_pt.stat().st_size / (1024 * 1024)
            dataset_total += size_mb
            total_cls_weights_mb += size_mb
            print(f"  {model:15s}: {size_mb:6.1f} MB")

    print(f"  {'':-<15s}  {'':-<8s}")
    print(f"  {'SUBTOTAL':15s}: {dataset_total:6.1f} MB")

print(f"\n{'='*24}")
print(f"TOTAL CLASSIFICATION MODEL WEIGHTS: {total_cls_weights_mb:6.1f} MB ({total_cls_weights_mb/1024:.2f} GB)")

# ============================================================================
# 2. DETECTION MODEL WEIGHTS
# ============================================================================
print("\n2. DETECTION MODEL WEIGHTS")
print("-" * 80)

total_det_weights_mb = 0

for dataset in datasets:
    print(f"\n{dataset.upper()}:")
    dataset_total = 0

    for model in det_models:
        model_dir = results_dir / f"experiment_{dataset}" / f"det_{model}" / "weights"
        best_pt = model_dir / "best.pt"

        if best_pt.exists():
            size_mb = best_pt.stat().st_size / (1024 * 1024)
            dataset_total += size_mb
            total_det_weights_mb += size_mb
            print(f"  {model:8s}: {size_mb:6.1f} MB")

    print(f"  {'':-<8s}  {'':-<8s}")
    print(f"  {'SUBTOTAL':8s}: {dataset_total:6.1f} MB")

print(f"\n{'='*18}")
print(f"TOTAL DETECTION MODEL WEIGHTS: {total_det_weights_mb:6.1f} MB ({total_det_weights_mb/1024:.2f} GB)")

# ============================================================================
# 3. TOTAL SHARED APPROACH STORAGE
# ============================================================================
print("\n3. TOTAL MODEL WEIGHTS STORAGE (Shared Approach)")
print("-" * 80)

total_shared_mb = total_cls_weights_mb + total_det_weights_mb
total_shared_gb = total_shared_mb / 1024

print(f"  Classification weights: {total_cls_weights_mb:7.1f} MB ({total_cls_weights_mb/1024:.2f} GB)")
print(f"  Detection weights:      {total_det_weights_mb:7.1f} MB ({total_det_weights_mb/1024:.2f} GB)")
print(f"  {'':-<24}  {'':-<12s}")
print(f"  TOTAL:                  {total_shared_mb:7.1f} MB ({total_shared_gb:.2f} GB)")

# ============================================================================
# 4. TRADITIONAL APPROACH ESTIMATE
# ============================================================================
print("\n4. TRADITIONAL APPROACH STORAGE ESTIMATE")
print("-" * 80)

print("\nTraditional approach stores:")
print("  - 3 detection models × 4 datasets = 12 detection weights (same as shared)")
print("  - 6 classification models × 3 detection models × 4 datasets = 72 cls weights")
print("")
print("Shared approach stores:")
print("  - 3 detection models × 4 datasets = 12 detection weights")
print("  - 6 classification models × 4 datasets = 24 cls weights")
print("")

# Calculate average classification weight size
avg_cls_weight_mb = total_cls_weights_mb / 24  # 24 cls models in shared

traditional_det_mb = total_det_weights_mb  # Same detection weights
traditional_cls_mb = avg_cls_weight_mb * 72  # 72 cls models in traditional
traditional_total_mb = traditional_det_mb + traditional_cls_mb
traditional_total_gb = traditional_total_mb / 1024

print(f"Average classification weight: {avg_cls_weight_mb:.1f} MB")
print(f"\nTraditional Approach Total:")
print(f"  Detection weights:      {traditional_det_mb:7.1f} MB (same as shared)")
print(f"  Classification weights: {traditional_cls_mb:7.1f} MB (72 models × {avg_cls_weight_mb:.1f} MB)")
print(f"  {'':-<24}  {'':-<12s}")
print(f"  TOTAL:                  {traditional_total_mb:7.1f} MB ({traditional_total_gb:.2f} GB)")

# ============================================================================
# 5. COMPARISON WITH PAPER CLAIMS
# ============================================================================
print("\n5. COMPARISON WITH PAPER CLAIMS")
print("="*80)

paper_shared_claim_mb = 600  # MB
paper_traditional_claim_mb = 1800  # MB (1.8 GB)

print(f"\nPAPER CLAIMS:")
print(f"  Shared approach:      {paper_shared_claim_mb:7.0f} MB ({paper_shared_claim_mb/1024:.2f} GB)")
print(f"  Traditional approach: {paper_traditional_claim_mb:7.0f} MB ({paper_traditional_claim_mb/1024:.2f} GB)")
print(f"  Reduction:            {(1 - paper_shared_claim_mb/paper_traditional_claim_mb)*100:.1f}%")

print(f"\nACTUAL DATA (Model Weights Only):")
print(f"  Shared approach:      {total_shared_mb:7.1f} MB ({total_shared_gb:.2f} GB)")
print(f"  Traditional approach: {traditional_total_mb:7.1f} MB ({traditional_total_gb:.2f} GB)")
print(f"  Reduction:            {(1 - total_shared_mb/traditional_total_mb)*100:.1f}%")

print(f"\nDISCREPANCY:")
print(f"  Shared claim vs actual:      {paper_shared_claim_mb:6.0f} MB vs {total_shared_mb:6.1f} MB = {abs(paper_shared_claim_mb - total_shared_mb):6.1f} MB off")
print(f"  Traditional claim vs actual: {paper_traditional_claim_mb:6.0f} MB vs {traditional_total_mb:6.1f} MB = {abs(paper_traditional_claim_mb - traditional_total_mb):6.1f} MB off")

# Error percentages
shared_error_pct = abs(paper_shared_claim_mb - total_shared_mb) / total_shared_mb * 100
traditional_error_pct = abs(paper_traditional_claim_mb - traditional_total_mb) / traditional_total_mb * 100

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

print(f"\n1. Shared Approach Storage (model weights): {total_shared_mb:.0f} MB ({total_shared_gb:.2f} GB)")
print(f"   Paper claims: {paper_shared_claim_mb} MB")
print(f"   Status: {'❌ NEEDS CORRECTION' if shared_error_pct > 20 else '✅ ACCEPTABLE'}")

print(f"\n2. Traditional Approach Storage (estimated): {traditional_total_mb:.0f} MB ({traditional_total_gb:.2f} GB)")
print(f"   Paper claims: {paper_traditional_claim_mb} MB (1.8 GB)")
print(f"   Status: {'❌ NEEDS CORRECTION' if traditional_error_pct > 20 else '✅ ACCEPTABLE'}")

print(f"\n3. Storage Reduction: {(1 - total_shared_mb/traditional_total_mb)*100:.1f}%")
print(f"   Paper claims: 67% reduction")

print("\nNOTE: This calculation includes ONLY model weights (.pt files).")
print("Total folder sizes are larger due to training logs, checkpoints, visualizations, etc.")

print("\n" + "="*80)
