# 📖 Manual Execution Guide - Ground Truth Crops & Classification Training

Guide lengkap untuk menjalankan crop generation dan classification training secara manual tanpa pipeline.

## 🎯 Overview

Workflow manual terdiri dari 2 tahap utama:
1. **Generate Ground Truth Crops**: Membuat dataset crops dari raw annotations
2. **Train Classification Models**: Melatih model PyTorch pada crops yang dihasilkan

## 📋 Prerequisites

### Environment Setup
```bash
# Pastikan Python environment sudah aktif
# GPU support (recommended)
nvidia-smi  # Check GPU availability
```

### Directory Structure
```
hello_world/
├── data/
│   ├── raw/                          # Raw datasets
│   │   ├── iml_lifecycle/            # IML lifecycle dataset
│   │   ├── mp_idb_species/           # MP-IDB species dataset
│   │   └── mp_idb_stages/            # MP-IDB stages dataset
│   └── ground_truth_crops_224/       # Generated crops (output)
├── scripts/training/
│   ├── generate_ground_truth_crops.py
│   └── 12_train_pytorch_classification.py
└── results/                          # Training results (output)
```

---

## 🌱 STAGE 1: Generate Ground Truth Crops

### 1.1 Basic Usage

#### IML Lifecycle Dataset (4 stages)
```bash
python scripts/training/generate_ground_truth_crops.py \
  --dataset "data/raw/iml_lifecycle" \
  --output "data/ground_truth_crops_224" \
  --type "iml_lifecycle" \
  --crop_size 224
```

#### MP-IDB Species Dataset (4 species)
```bash
python scripts/training/generate_ground_truth_crops.py \
  --dataset "data/raw/mp_idb_species" \
  --output "data/ground_truth_crops_224" \
  --type "mp_idb_species" \
  --crop_size 224
```

#### MP-IDB Stages Dataset (4 stages)
```bash
python scripts/training/generate_ground_truth_crops.py \
  --dataset "data/raw/mp_idb_stages" \
  --output "data/ground_truth_crops_224" \
  --type "mp_idb_stages" \
  --crop_size 224
```

### 1.2 Advanced Options

#### Custom Parameters
```bash
python scripts/training/generate_ground_truth_crops.py \
  --dataset "data/raw/iml_lifecycle" \
  --output "data/ground_truth_crops_custom" \
  --type "iml_lifecycle" \
  --crop_size 256 \
  --padding 20 \
  --min_size 50
```

#### Available Parameters
- `--dataset`: Path ke raw dataset
- `--output`: Output directory untuk crops
- `--type`: Dataset type (`iml_lifecycle`, `mp_idb_species`, `mp_idb_stages`)
- `--crop_size`: Size crops dalam pixels (default: 224)
- `--padding`: Extra padding around bounding box (default: 10)
- `--min_size`: Minimum crop size untuk filtering (default: 32)

### 1.3 Expected Output

#### Directory Structure Created
```
data/ground_truth_crops_224/
└── lifecycle/                        # or species/stages
    ├── train/
    │   ├── gametocyte/               # Class folders
    │   ├── ring/
    │   ├── schizont/
    │   └── trophozoite/
    ├── val/
    │   ├── gametocyte/
    │   ├── ring/
    │   ├── schizont/
    │   └── trophozoite/
    └── test/
        ├── gametocyte/
        ├── ring/
        ├── schizont/
        └── trophozoite/
```

#### Sample Output Log
```
[INFO] Processing IML lifecycle dataset
[INFO] Found 310 images with annotations
[STRATIFY] Overall class distribution: {0: 120, 1: 95, 2: 15, 3: 80}
[STRATIFY] Split: 217 train, 62 val, 31 test
[INFO] Generated 1,845 total crops
[INFO] Train: 1,291 crops, Val: 369 crops, Test: 185 crops
[SUCCESS] Ground truth crops generated successfully!
```

#### Quality Checks
- ✅ **Stratified Splits**: 70% train, 20% val, 10% test
- ✅ **Class Balance**: All classes represented in each split
- ✅ **Deterministic**: Same results with same random_state=42
- ✅ **No Duplicates**: Fixed duplicate generation bug

---

## 🚀 STAGE 2: Train Classification Models

### 2.1 Basic Training Commands

#### Single Model Training
```bash
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "efficientnet_b0" \
  --loss "focal" \
  --epochs 30 \
  --batch 32 \
  --lr 0.0005 \
  --focal_alpha 1.0 \
  --focal_gamma 2.0 \
  --name "lifecycle_efficientnet_focal" \
  --save-dir "results/manual_training"
```

#### Cross-Entropy Training (Baseline)
```bash
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "efficientnet_b0" \
  --loss "cross_entropy" \
  --epochs 30 \
  --batch 32 \
  --lr 0.001 \
  --name "lifecycle_efficientnet_ce" \
  --save-dir "results/manual_training"
```

### 2.2 Multiple Model Training

#### All Baseline Models
```bash
# EfficientNet-B0 (Recommended)
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "efficientnet_b0" \
  --loss "focal" \
  --epochs 30 \
  --name "lifecycle_efficientnet_b0_focal"

# DenseNet121 (Dense connections)
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "densenet121" \
  --loss "focal" \
  --epochs 30 \
  --name "lifecycle_densenet121_focal"

# ResNet18 (Lightweight)
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "resnet18" \
  --loss "focal" \
  --epochs 30 \
  --name "lifecycle_resnet18_focal"

# ConvNeXt-Tiny (Modern architecture)
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "convnext_tiny" \
  --loss "focal" \
  --epochs 30 \
  --name "lifecycle_convnext_focal"
```

#### Advanced Models
```bash
# EfficientNet-B1 (Larger)
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "efficientnet_b1" \
  --loss "focal" \
  --epochs 25 \
  --batch 16 \
  --lr 0.0003 \
  --name "lifecycle_efficientnet_b1_focal"

# EfficientNet-B2 (Even larger)
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "efficientnet_b2" \
  --loss "focal" \
  --epochs 25 \
  --batch 16 \
  --lr 0.0003 \
  --name "lifecycle_efficientnet_b2_focal"

# ResNet101 (Deep)
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "resnet101" \
  --loss "focal" \
  --epochs 30 \
  --batch 16 \
  --name "lifecycle_resnet101_focal"

# MobileNet-V3-Large (Mobile-optimized)
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "mobilenet_v3_large" \
  --loss "focal" \
  --epochs 30 \
  --name "lifecycle_mobilenet_focal"
```

### 2.3 Parameter Configuration

#### Optimal Parameters (Based on Experiments)

**For Medical Data (Recommended):**
```bash
--loss "focal"
--focal_alpha 1.0      # Balanced class weighting
--focal_gamma 2.0      # Hard example focus
--lr 0.0005            # Stable learning rate
--batch 32             # For 224px images
--epochs 30            # Sufficient for convergence
```

**For Baseline Comparison:**
```bash
--loss "cross_entropy"
--lr 0.001             # Higher LR for CE
--batch 32
--epochs 30
```

#### Model-Specific Tuning
```bash
# Large models (B1, B2, ResNet101)
--batch 16             # Reduce batch size
--lr 0.0003            # Lower learning rate
--epochs 25            # Fewer epochs (prevent overfitting)

# Lightweight models (ResNet18, MobileNet)
--batch 32             # Standard batch size
--lr 0.0005            # Standard learning rate
--epochs 30            # Standard epochs
```

### 2.4 Available Parameters

#### Essential Parameters
- `--data`: Path to crops dataset
- `--model`: Model architecture name
- `--loss`: Loss function (`focal` or `cross_entropy`)
- `--epochs`: Number of training epochs
- `--name`: Experiment name
- `--save-dir`: Output directory

#### Advanced Parameters
- `--batch`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.0005)
- `--focal_alpha`: Focal loss alpha parameter (default: 1.0)
- `--focal_gamma`: Focal loss gamma parameter (default: 2.0)
- `--device`: Device (`cuda` or `cpu`)
- `--num_workers`: DataLoader workers (default: 4)

#### Supported Models
- `resnet18`, `resnet101`
- `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
- `densenet121`
- `convnext_tiny`
- `mobilenet_v3_large`

---

## 📊 Expected Results & Performance

### 2.5 Training Output

#### Sample Training Log
```
Model: efficientnet_b0
Loss: focal (α=1.0, γ=2.0)
Dataset: lifecycle (58 test samples)

Epoch [25/30] Train Loss: 0.234 | Train Acc: 92.45%
Epoch [25/30] Val Loss: 0.298   | Val Acc: 89.23%

Best Val Acc: 91.47%
Test Acc: 88.24%
Balanced Acc: 88.67%
Training Time: 2.1 min

Classification Report:
              precision    recall  f1-score   support
  gametocyte       0.90      0.97      0.93        30
        ring       0.94      0.82      0.88        17
    schizont       1.00      0.67      0.80         3
 trophozoite       0.75      0.75      0.75         8

    accuracy                           0.88        58
   macro avg       0.90      0.80      0.84        58
weighted avg       0.89      0.88      0.88        58
```

#### Output Files Generated
```
results/manual_training/lifecycle_efficientnet_b0_focal/
├── model_best.pth                    # Best model weights
├── results.txt                       # Performance summary
├── classification_report.txt         # Detailed metrics
├── confusion_matrix.png             # Confusion matrix plot
├── training_curves.png              # Loss/accuracy curves
└── config.json                      # Training configuration
```

### 2.6 Performance Benchmarks

#### Expected Accuracies (IML Lifecycle)

**Focal Loss (Medical-Optimized):**
- ✅ EfficientNet-B0: **88-92%** (Recommended)
- ✅ DenseNet121: **85-89%**
- ✅ ResNet18: **82-86%** (Lightweight)
- ✅ ConvNeXt-Tiny: **86-90%**

**Cross-Entropy (Baseline):**
- ⚠️ EfficientNet-B0: **84-88%** (Lower than focal)
- ⚠️ DenseNet121: **82-86%**
- ⚠️ ResNet18: **78-82%**

**Overfitting Risk:**
- ❌ EfficientNet-B1: **74-78%** (Too large, overfits)
- ❌ EfficientNet-B2: **70-75%** (Severe overfitting)

---

## 🔄 Complete Workflow Examples

### 3.1 Quick Start (Single Dataset)

```bash
# Step 1: Generate crops for IML lifecycle
python scripts/training/generate_ground_truth_crops.py \
  --dataset "data/raw/iml_lifecycle" \
  --output "data/ground_truth_crops_224" \
  --type "iml_lifecycle" \
  --crop_size 224

# Step 2: Train best model
python scripts/training/12_train_pytorch_classification.py \
  --data "data/ground_truth_crops_224/lifecycle" \
  --model "efficientnet_b0" \
  --loss "focal" \
  --epochs 30 \
  --name "lifecycle_best_model" \
  --save-dir "results/manual_training"
```

### 3.2 Multi-Dataset Training

```bash
# Generate crops for all datasets
python scripts/training/generate_ground_truth_crops.py \
  --dataset "data/raw/iml_lifecycle" \
  --output "data/ground_truth_crops_224" \
  --type "iml_lifecycle" \
  --crop_size 224

python scripts/training/generate_ground_truth_crops.py \
  --dataset "data/raw/mp_idb_species" \
  --output "data/ground_truth_crops_224" \
  --type "mp_idb_species" \
  --crop_size 224

python scripts/training/generate_ground_truth_crops.py \
  --dataset "data/raw/mp_idb_stages" \
  --output "data/ground_truth_crops_224" \
  --type "mp_idb_stages" \
  --crop_size 224

# Train on all datasets
for dataset in lifecycle species stages; do
  python scripts/training/12_train_pytorch_classification.py \
    --data "data/ground_truth_crops_224/$dataset" \
    --model "efficientnet_b0" \
    --loss "focal" \
    --epochs 30 \
    --name "${dataset}_efficientnet_focal" \
    --save-dir "results/manual_training"
done
```

### 3.3 Systematic Model Comparison

```bash
# Compare multiple models on lifecycle dataset
models=("resnet18" "efficientnet_b0" "densenet121" "convnext_tiny")
losses=("focal" "cross_entropy")

for model in "${models[@]}"; do
  for loss in "${losses[@]}"; do
    echo "Training $model with $loss loss..."

    if [ "$loss" = "focal" ]; then
      lr=0.0005
      extra_args="--focal_alpha 1.0 --focal_gamma 2.0"
    else
      lr=0.001
      extra_args=""
    fi

    python scripts/training/12_train_pytorch_classification.py \
      --data "data/ground_truth_crops_224/lifecycle" \
      --model "$model" \
      --loss "$loss" \
      --lr "$lr" \
      --epochs 30 \
      --name "lifecycle_${model}_${loss}" \
      --save-dir "results/systematic_comparison" \
      $extra_args
  done
done
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
--batch 16

# For very large models
--batch 8
```

#### 2. Dataset Not Found
```bash
# Check directory structure
ls -la data/raw/
ls -la data/ground_truth_crops_224/

# Regenerate crops if missing
python scripts/training/generate_ground_truth_crops.py ...
```

#### 3. Low Accuracy
```bash
# Use focal loss for medical data
--loss "focal" --focal_alpha 1.0 --focal_gamma 2.0

# Try different learning rate
--lr 0.0005  # For focal loss
--lr 0.001   # For cross-entropy

# Increase epochs
--epochs 50
```

#### 4. Overfitting
```bash
# Use smaller model
--model "resnet18"  # Instead of efficientnet_b1

# Reduce learning rate
--lr 0.0003

# Fewer epochs
--epochs 25
```

### Performance Validation

#### Expected File Sizes
- **Lifecycle crops**: ~1,800 images total
- **Species crops**: ~2,500 images total
- **Stages crops**: ~2,200 images total

#### Test Set Sizes (with crop_size=224)
- **Lifecycle**: 58 test samples
- **Species**: ~75 test samples
- **Stages**: ~65 test samples

---

## 🏁 Summary

### Manual vs Pipeline
- ✅ **Manual**: Fine-grained control, custom parameters
- ✅ **Pipeline**: Automated, systematic comparison
- ✅ **Both**: Use ground truth crops (not detection-based)

### Best Practices
1. **Always use focal loss** for medical data
2. **Start with EfficientNet-B0** for best balance
3. **Use crop_size=224** for consistency
4. **Monitor balanced accuracy** for medical applications
5. **Generate crops once**, train multiple models

### Performance Expectations
- 🎯 **Target Accuracy**: >85% for lifecycle classification
- 🏆 **Best Model**: EfficientNet-B0 + Focal Loss
- ⚡ **Training Time**: 2-5 minutes per model (RTX 3060)
- 📊 **Balanced Accuracy**: Critical for medical diagnosis

---

*Last Updated: 2025-09-28*
*Guide Version: 1.0 - Manual Execution Workflow*