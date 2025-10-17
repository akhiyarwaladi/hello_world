# SOLUSI SEBENARNYA: K-Fold Cross Validation

## 🔬 ROOT CAUSE ANALYSIS

### Anda BENAR - Sudah Ada Augmentation & Weighted Sampling!

**Current System:**
1. ✅ **Augmentation** - RandomFlip, Rotation, ColorJitter, MildMedicalAugmentation
2. ✅ **Weighted Random Sampler** - Minority class di-sample 8.58x lebih sering!
3. ✅ **Focal Loss** (α=1.0, γ=1.5) - Fokus ke hard examples

**Data Distribution:**

| Split | P_falciparum | P_malariae | P_ovale | P_vivax | Total |
|-------|--------------|------------|---------|---------|-------|
| **Train** | 773 (90.1%) | 25 (2.9%) | 19 (2.2%) | 41 (4.8%) | 858 |
| **Val** | 314 (91.5%) | 10 (2.9%) | 8 (2.3%) | 11 (3.2%) | 343 |
| **Test** | 210 (89.4%) | 8 (3.4%) | **5 (2.1%)** | 12 (5.1%) | 235 |

### MASALAH SEBENARNYA: Test Set Terlalu Kecil!

**Test Results P_ovale (5 samples total):**
- EfficientNet-B0: 5/5 (100%) ✅
- ResNet101: 4/5 (80%) ❌ - **1 error = -20% accuracy!**
- ResNet50: 3/5 (60%) ❌ - **2 errors = -40% accuracy!**

**Impact:**
- 1 kesalahan klasifikasi = -20% accuracy!
- 2 kesalahan = -40% accuracy!
- **HIGH VARIANCE** - tidak reliable untuk evaluasi!

---

## ⭐⭐⭐⭐⭐ SOLUSI: 5-FOLD CROSS VALIDATION

### Kenapa K-Fold?

**Problem:** Test set 5-12 samples per class = tidak cukup untuk reliable evaluation

**Solution:** K-Fold Cross Validation
- Setiap fold punya test set berbeda
- 5 folds = 5 evaluations dengan test sets berbeda
- Report: **Mean ± Std** (e.g., 98.3% ± 1.2%)

**Benefits:**
1. ✅ **More reliable metrics** - Average dari 5 runs
2. ✅ **Confidence interval** - Std deviation menunjukkan variance
3. ✅ **Use all data** - Semua samples dipakai untuk train & test
4. ✅ **Better for small datasets** - Maximize data utilization

---

## 🛠️ IMPLEMENTATION

### Option A: Stratified K-Fold (Recommended)

Buat script baru: `scripts/training/kfold_classification.py`

```python
#!/usr/bin/env python3
"""
K-Fold Cross Validation for Classification Models
Provides confidence intervals for small datasets
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import json

def run_kfold_cv(model_name, crops_path, n_splits=5, epochs=75):
    """
    Run K-Fold Cross Validation for classification

    Args:
        model_name: Model architecture (e.g., 'efficientnet_b0')
        crops_path: Path to crops folder with train+val+test combined
        n_splits: Number of folds (default: 5)
        epochs: Training epochs per fold

    Returns:
        Dict with mean ± std metrics
    """

    # Combine all samples (train + val + test)
    all_samples = []
    all_labels = []

    for class_idx, class_folder in enumerate(sorted(crops_path.iterdir())):
        if class_folder.is_dir():
            class_samples = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
            all_samples.extend(class_samples)
            all_labels.extend([class_idx] * len(class_samples))

    print(f"[K-FOLD] Total samples: {len(all_samples)}")
    print(f"[K-FOLD] Classes: {len(set(all_labels))}")

    # Stratified K-Fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store results for each fold
    fold_results = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(all_samples, all_labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*60}")

        # Further split train_val into train (80%) and val (20%)
        train_val_samples = [all_samples[i] for i in train_val_idx]
        train_val_labels = [all_labels[i] for i in train_val_idx]

        # Split train_val into train and val
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            range(len(train_val_samples)),
            test_size=0.2,
            stratify=train_val_labels,
            random_state=42
        )

        train_samples = [train_val_samples[i] for i in train_idx]
        val_samples = [train_val_samples[i] for i in val_idx]
        test_samples = [all_samples[i] for i in test_idx]

        print(f"[FOLD {fold_idx+1}] Train: {len(train_samples)}")
        print(f"[FOLD {fold_idx+1}] Val: {len(val_samples)}")
        print(f"[FOLD {fold_idx+1}] Test: {len(test_samples)}")

        # Create temporary train/val/test folders for this fold
        fold_dir = Path(f'temp_kfold/fold_{fold_idx}')
        create_fold_dataset(fold_dir, train_samples, val_samples, test_samples, all_labels)

        # Train model on this fold
        from scripts.training.12_train_pytorch_classification import main as train_main

        # Call training with fold-specific data
        import sys
        sys.argv = [
            'train',
            '--data', str(fold_dir),
            '--model', model_name,
            '--epochs', str(epochs),
            '--batch', '64',
            '--lr', '0.0005',
            '--loss', 'focal',
            '--focal_alpha', '1.0',
            '--focal_gamma', '1.5',
            '--save-dir', str(fold_dir / 'results')
        ]

        train_main()

        # Load results
        results_path = fold_dir / 'results' / 'table9_metrics.json'
        with open(results_path) as f:
            metrics = json.load(f)

        fold_results.append({
            'fold': fold_idx + 1,
            'test_accuracy': metrics['test_accuracy'],
            'balanced_accuracy': metrics['overall_balanced_accuracy'],
            'per_class': metrics['per_class_metrics']
        })

        print(f"\n[FOLD {fold_idx+1}] Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
        print(f"[FOLD {fold_idx+1}] Balanced Accuracy: {metrics['overall_balanced_accuracy']*100:.2f}%")

    # Calculate statistics across folds
    test_accs = [r['test_accuracy'] for r in fold_results]
    balanced_accs = [r['balanced_accuracy'] for r in fold_results]

    results_summary = {
        'model': model_name,
        'n_folds': n_splits,
        'test_accuracy_mean': np.mean(test_accs),
        'test_accuracy_std': np.std(test_accs),
        'balanced_accuracy_mean': np.mean(balanced_accs),
        'balanced_accuracy_std': np.std(balanced_accs),
        'fold_results': fold_results
    }

    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS VALIDATION RESULTS ({n_splits} folds)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Test Accuracy: {results_summary['test_accuracy_mean']*100:.2f}% ± {results_summary['test_accuracy_std']*100:.2f}%")
    print(f"Balanced Accuracy: {results_summary['balanced_accuracy_mean']*100:.2f}% ± {results_summary['balanced_accuracy_std']*100:.2f}%")

    # Save summary
    with open(f'kfold_results_{model_name}.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    return results_summary


def create_fold_dataset(fold_dir, train_samples, val_samples, test_samples, all_labels):
    """Create train/val/test folders for this fold"""
    import shutil

    # Create directories
    for split in ['train', 'val', 'test']:
        split_dir = fold_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create class subdirectories
        unique_classes = sorted(set(all_labels))
        for class_idx in unique_classes:
            (split_dir / f'class_{class_idx}').mkdir(exist_ok=True)

    # Copy files to appropriate folders
    def copy_samples(samples, split_name):
        for sample_path in samples:
            # Determine class from original path
            class_name = sample_path.parent.name
            dest_path = fold_dir / split_name / class_name / sample_path.name
            shutil.copy(sample_path, dest_path)

    copy_samples(train_samples, 'train')
    copy_samples(val_samples, 'val')
    copy_samples(test_samples, 'test')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K-Fold Cross Validation")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--data", required=True, help="Path to combined crops")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=75, help="Epochs per fold")

    args = parser.parse_args()

    crops_path = Path(args.data)
    run_kfold_cv(args.model, crops_path, args.folds, args.epochs)
```

### Usage:

```bash
# Combine all crops first (train + val + test into one folder)
python scripts/utils/combine_crops.py \
  --input results/optA_20251016_200330/experiments/experiment_mp_idb_species/crops_gt_crops/crops \
  --output data/combined_crops_mp_idb_species

# Run 5-fold CV for EfficientNet-B0
python scripts/training/kfold_classification.py \
  --model efficientnet_b0 \
  --data data/combined_crops_mp_idb_species \
  --folds 5 \
  --epochs 75

# Run for ResNet101 to compare
python scripts/training/kfold_classification.py \
  --model resnet101 \
  --data data/combined_crops_mp_idb_species \
  --folds 5 \
  --epochs 75
```

---

## 📊 EXPECTED RESULTS dengan K-Fold

### Current (Single Test Set):
- EfficientNet-B0: **98.30%** test accuracy
- ResNet101: **93.19%** test accuracy
- ResNet50: **88.51%** test accuracy

### With 5-Fold CV (Expected):
- EfficientNet-B0: **98.5% ± 0.8%** (95% CI: 97.7-99.3%)
- ResNet101: **94.0% ± 2.5%** (95% CI: 91.5-96.5%)
- ResNet50: **90.0% ± 3.2%** (95% CI: 86.8-93.2%)

**Benefits:**
1. ✅ **Confidence interval** - Know the variance!
2. ✅ **More reliable** - Average dari 5 runs
3. ✅ **Better comparison** - Dapat compare model dengan statistical significance
4. ✅ **For publication** - Journals prefer K-Fold untuk small datasets

---

## 🎯 ALTERNATIVE: Increase Test Set Size

Jika tidak mau K-Fold (karena 5x training time), bisa **adjust split ratio:**

### Current Split (60/24/16):
- Train: 858 (60%)
- Val: 343 (24%)
- Test: 235 (16%) ← P_ovale hanya 5 samples!

### Recommended Split (50/20/30):
- Train: 715 (50%)
- Val: 287 (20%)
- Test: **430 (30%)** ← P_ovale akan ~10-15 samples!

**Change in `main_pipeline.py`:**

```python
parser.add_argument("--train-ratio", type=float, default=0.50,  # Changed from 0.60
                   help="Training set ratio (default: 0.50 = 50%)")
parser.add_argument("--val-ratio", type=float, default=0.20,  # Changed from 0.24
                   help="Validation set ratio (default: 0.20 = 20%)")
parser.add_argument("--test-ratio", type=float, default=0.30,  # Changed from 0.16
                   help="Test set ratio (default: 0.30 = 30%)")
```

**Expected dengan 30% test:**
- P_malariae: 8 → **15-16 samples** (+88%)
- P_ovale: 5 → **10-12 samples** (+100%)
- P_vivax: 12 → **22-24 samples** (+100%)

**Impact pada variance:**
- 1 error di P_ovale: 20% → **10%** (50% reduction!)
- 2 errors di P_ovale: 40% → **20%** (50% reduction!)

---

## ✅ FINAL RECOMMENDATION

### **Option 1: K-Fold CV** (Best for Science)
- **Pros:** Most reliable, confidence intervals, accepted by journals
- **Cons:** 5x training time (5 days for all models)
- **Use when:** Publishing in top journals, need statistical rigor

### **Option 2: Increase Test Set (50/20/30)** (Quick Fix)
- **Pros:** Simple, 1 day to re-run, reduce variance
- **Cons:** Less reliable than K-Fold, still some variance
- **Use when:** Quick improvement, limited time

### **Option 3: Accept Current Results** (Pragmatic)
- EfficientNet-B0 **98.30%** is EXCELLENT
- Write in paper: "Limited by small minority class samples (5-12 per class)"
- Focus on EfficientNet-B0 success story

---

## 💡 BOTTOM LINE

**Anda BENAR sekali!**

Sistem SUDAH:
- ✅ Augmentation (on-the-fly)
- ✅ Weighted sampling (minority 8x more frequent)
- ✅ Focal loss (focus on hard examples)

**Masalahnya BUKAN training, tapi EVALUATION:**
- Test set terlalu kecil (5-12 samples per minority class)
- 1-2 kesalahan = drop 20-40% accuracy!
- **High variance** dalam evaluation metrics

**Solusi:**
1. **K-Fold CV** - Most robust
2. **Increase test set** - Quick fix (50/20/30 split)
3. **Accept & publish** - 98.3% sudah excellent

**Mana yang mau dicoba?**
