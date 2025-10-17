# OPSI PENINGKATAN AKURASI - Action Plan

## 🎯 MASALAH YANG DITEMUKAN

**Data Analysis:**
- EfficientNet-B0: 98.30% test (gap 0.83%) ✅ **EXCELLENT - Generalisasi baik**
- DenseNet121: 96.17% test (gap 2.96%) ✅ Good
- ResNet101: 93.19% test (gap 5.94%) ❌ **Overfitting**
- ResNet50: 88.51% test (gap 10.62%) ❌ **Severe Overfitting**

**Root Cause:**
- ResNet50/101 terlalu dalam (25M-44M parameters) untuk dataset kecil (1,436 samples)
- Ratio params/sample: ResNet50 = 17,421:1 vs EfficientNet-B0 = 3,482:1

---

## 📋 OPSI PRAKTIS (Ranked by Effectiveness)

### ⭐⭐⭐⭐⭐ **OPSI 1: REGULARIZATION TUNING** (Paling Mudah & Efektif)

**Target:** ResNet50: 88.51% → 91-93%, ResNet101: 93.19% → 95-96%

**Expected Time:** 1-2 hari (re-train ResNet models saja)

**Expected Success Rate:** 70%

#### **Perubahan 1: Tambah Dropout ke ResNet**

File: `scripts/training/12_train_pytorch_classification.py`

**Cari baris 107-120 (ResNet model definition):**
```python
if model_name.startswith('resnet'):
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    elif model_name == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
    elif model_name == 'resnet101':
        model = models.resnet101(weights='IMAGENET1K_V2' if pretrained else None)
    else:
        raise ValueError(f"Unknown ResNet model: {model_name}")

    # Modify final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
```

**GANTI DENGAN (tambah dropout 0.5):**
```python
if model_name.startswith('resnet'):
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    elif model_name == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
    elif model_name == 'resnet101':
        model = models.resnet101(weights='IMAGENET1K_V2' if pretrained else None)
    else:
        raise ValueError(f"Unknown ResNet model: {model_name}")

    # IMPROVED: Add dropout to reduce overfitting on small datasets
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Strong regularization for ResNet overfitting
        nn.Linear(in_features, num_classes)
    )
```

#### **Perubahan 2: Tingkatkan Weight Decay**

**Cari baris 672 (optimizer definition):**
```python
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
```

**GANTI DENGAN:**
```python
# IMPROVED: Stronger weight decay for ResNet models (reduce overfitting)
weight_decay = 5e-4 if args.model.startswith('resnet') else 1e-4
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
print(f"[REGULARIZATION] Weight decay: {weight_decay} ({'ResNet' if args.model.startswith('resnet') else 'Others'})")
```

#### **Perubahan 3: Label Smoothing**

**Tambahkan parameter label smoothing ke FocalLoss:**

**Cari baris 69-102 (FocalLoss class):**
```python
class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance"""

    def __init__(self, alpha=1.0, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # ... rest of code
```

**GANTI DENGAN (tambah label_smoothing):**
```python
class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance

    Reference: https://arxiv.org/abs/1708.02002
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 1.5)
        reduction (str): Specifies reduction to apply to output
        label_smoothing (float): Label smoothing for regularization (0.0-0.2)
    """

    def __init__(self, alpha=1.0, gamma=1.5, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Calculate cross entropy with label smoothing
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                 label_smoothing=self.label_smoothing)

        # Calculate p_t
        pt = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

**Dan update baris 640 (criterion initialization):**
```python
criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, label_smoothing=0.1)
print(f"[FOCAL] Alpha: {args.focal_alpha}, Gamma: {args.focal_gamma}, Label Smoothing: 0.1")
```

---

### ⭐⭐⭐⭐ **OPSI 2: TUNING FOCAL LOSS PARAMETERS**

**Target:** EfficientNet-B0: 98.30% → 98.5-99.0%

**Expected Time:** 1 hari (re-train dengan alpha berbeda)

**Expected Success Rate:** 60%

#### **Cara Testing:**

Run experiment dengan alpha berbeda:

```bash
# Alpha 0.25 (standard for minority class focus)
python main_pipeline.py --dataset mp_idb_species --include yolo11 --classification-models efficientnet_b0 --focal-alpha 0.25 --focal-gamma 2.0

# Alpha 0.5 (stronger minority class focus)
python main_pipeline.py --dataset mp_idb_species --include yolo11 --classification-models efficientnet_b0 --focal-alpha 0.5 --focal-gamma 2.0

# Alpha 0.75 (very strong minority class focus)
python main_pipeline.py --dataset mp_idb_species --include yolo11 --classification-models efficientnet_b0 --focal-alpha 0.75 --focal-gamma 2.0
```

**Catatan:** Perlu tambahkan `--focal-alpha` parameter ke main_pipeline.py argument parser

---

### ⭐⭐⭐ **OPSI 3: TINGKATKAN EPOCHS**

**Target:** Semua model: +1-2% improvement

**Expected Time:** 2-3 hari (lebih lama training)

**Expected Success Rate:** 50%

#### **Perubahan:**

File: `main_pipeline.py`

**Cari default epochs classification (sekarang 75):**
```python
parser.add_argument("--epochs-cls", type=int, default=75,
                   help="Classification training epochs (default: 75)")
```

**GANTI DENGAN:**
```python
parser.add_argument("--epochs-cls", type=int, default=100,
                   help="Classification training epochs (default: 100, increased for better convergence)")
```

**Jalankan:**
```bash
python main_pipeline.py --dataset mp_idb_species --epochs-cls 100
```

---

### ⭐⭐⭐ **OPSI 4: TEST TIME AUGMENTATION (TTA)**

**Target:** +0.5-1.5% improvement tanpa re-training

**Expected Time:** 2 jam (hanya inference)

**Expected Success Rate:** 80%

#### **Implementasi:**

Buat file baru: `scripts/inference/test_time_augmentation.py`

```python
#!/usr/bin/env python3
"""
Test Time Augmentation for Classification Models
Apply multiple augmentations at test time and average predictions
"""

import torch
from torchvision import transforms
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def get_tta_transforms(image_size=224):
    """Get list of test time augmentation transforms"""
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tta_transforms = [
        # Original
        base_transform,
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Vertical flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Both flips
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Slight rotation +5°
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation((5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Slight rotation -5°
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation((-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]

    return tta_transforms

def tta_predict(model, image, transforms_list, device):
    """Apply TTA and return averaged predictions"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for transform in transforms_list:
            # Transform image
            img_tensor = transform(image).unsqueeze(0).to(device)

            # Predict
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())

    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred

def evaluate_with_tta(model_path, test_data_path, device='cuda'):
    """Evaluate model with TTA"""
    from torchvision import datasets
    from scripts.training.12_train_pytorch_classification import get_model

    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_name = checkpoint['model_name']
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    model = get_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = datasets.ImageFolder(
        root=test_data_path,
        transform=None  # We'll apply transforms in TTA function
    )

    # Get TTA transforms
    tta_transforms = get_tta_transforms()

    # Evaluate
    correct = 0
    total = 0

    print(f"[TTA] Evaluating with {len(tta_transforms)} augmentations...")
    for img, label in tqdm(test_dataset):
        # TTA prediction
        pred_probs = tta_predict(model, img, tta_transforms, device)
        pred_class = np.argmax(pred_probs)

        if pred_class == label:
            correct += 1
        total += 1

    accuracy = 100 * correct / total
    print(f"\n[TTA] Test Accuracy: {accuracy:.2f}%")
    print(f"[TTA] Correct: {correct}/{total}")

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Time Augmentation Evaluation")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--data", required=True, help="Path to test data folder")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    accuracy = evaluate_with_tta(args.model, args.data, args.device)
```

**Cara pakai:**
```bash
# Test model dengan TTA
python scripts/inference/test_time_augmentation.py \
  --model results/optA_20251016_200330/experiments/experiment_mp_idb_species/cls_efficientnet_b0_focal/best.pt \
  --data results/optA_20251016_200330/experiments/experiment_mp_idb_species/crops_gt_crops/test
```

---

### ⭐⭐ **OPSI 5: FREEZE EARLY LAYERS (Transfer Learning Strategy)**

**Target:** ResNet: +2-3% improvement

**Expected Time:** 1 hari

**Expected Success Rate:** 50%

#### **Implementasi:**

Tambahkan di `12_train_pytorch_classification.py` setelah model initialization (baris ~596):

```python
# Initialize model
print(f"\n[LOAD] Loading {args.model} model...")
model = get_model(args.model, num_classes, args.pretrained)

# IMPROVED: Freeze early layers for ResNet to reduce overfitting
if args.model.startswith('resnet'):
    # Freeze all layers except final 2 blocks + FC layer
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[FREEZE] Froze early layers for ResNet")
    print(f"[FREEZE] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

model = model.to(device)
```

---

## 🎯 REKOMENDASI URUTAN EKSEKUSI

### **Week 1: Quick Wins (Opsi 1 + Opsi 4)**

1. **Hari 1-2:** Implementasi OPSI 1 (Regularization)
   - Edit `12_train_pytorch_classification.py`
   - Train ResNet50 & ResNet101 saja
   - Expected: ResNet101 93.19% → 95-96%

2. **Hari 3:** Implementasi OPSI 4 (TTA)
   - Buat `test_time_augmentation.py`
   - Test semua model yang sudah ada
   - Expected: EfficientNet-B0 98.30% → 98.8-99.0%

3. **Hari 4-5:** Jika hasil masih kurang, lanjut OPSI 2 (Focal Loss tuning)

**Total Expected Improvement:**
- ResNet101: 93.19% → **95-96%** (+2-3%)
- EfficientNet-B0: 98.30% → **98.8-99.0%** (+0.5-0.7% via TTA)

### **Week 2: Advanced (jika masih perlu)**

- OPSI 3: Tingkatkan epochs ke 100
- OPSI 5: Freeze layers untuk ResNet

---

## 📊 TRACKING RESULTS

Buat tabel tracking:

| Opsi | Model | Baseline | After | Improvement | Time | Notes |
|------|-------|----------|-------|-------------|------|-------|
| Baseline | EfficientNet-B0 | 98.30% | - | - | - | Best model |
| Baseline | ResNet101 | 93.19% | - | - | - | Overfitting |
| Baseline | ResNet50 | 88.51% | - | - | - | Severe overfit |
| Opsi 1 | ResNet101 | 93.19% | ??? | ??? | 1d | Regularization |
| Opsi 1 | ResNet50 | 88.51% | ??? | ??? | 1d | Regularization |
| Opsi 4 | EfficientNet-B0 | 98.30% | ??? | ??? | 2h | TTA |
| Opsi 2 | EfficientNet-B0 | 98.30% | ??? | ??? | 1d | Focal α=0.25 |

---

## ✅ NEXT STEPS

**Pilih salah satu:**

1. **[RECOMMENDED] Start with Opsi 1 + Opsi 4** - Paling praktis, ROI tertinggi
2. **Try all options** - Jika punya waktu 1-2 minggu
3. **Accept 98.3% and publish** - Jika deadline ketat (see ultrathink_analysis.md)

**Mana yang mau dicoba dulu?**

Saya bisa langsung implement Opsi 1 (Regularization) kalau Anda mau. Tinggal edit `12_train_pytorch_classification.py` dan re-run pipeline untuk ResNet models saja.
