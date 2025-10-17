# SOLUSI MINORITY CLASS - True Root Cause

## 🔬 ANALISA MENDALAM

### Masalah SEBENARNYA bukan ResNet Overfitting!

**DATA EVIDENCE:**

| Class | Test Samples | % | EfficientNet-B0 | ResNet101 | ResNet50 |
|-------|--------------|---|-----------------|-----------|----------|
| **P_falciparum** | 210 | 89.4% | 99.52% precision ✅ | 99.50% ✅ | 98.47% ✅ |
| **P_malariae** | 8 | 3.4% | 77.78% ✅ | 58.33% ❌ | 26.09% ❌ |
| **P_ovale** | 5 | 2.1% | 100% ✅ | 57.14% ❌ | 37.50% ❌ |
| **P_vivax** | 12 | 5.1% | 92.31% ✅ | 56.25% ❌ | 75.00% ⚠️ |

### KEY INSIGHT:

**ResNet TIDAK overfitting!** ResNet EXCELLENT di P_falciparum (98-99% precision)!

**Masalahnya:** ResNet FAILED di minority classes (8+5+12 = 25 samples total)

**Root Cause:**
1. **Extreme class imbalance:** 89.4% vs 10.6%
2. **Tiny minority samples:** 5-12 samples per class di test set
3. **ResNet "lazy learning":** Prediksi P_falciparum aja = 89.4% accuracy!
4. **EfficientNet-B0 menang** karena Focal Loss + parameter efficiency

---

## ⭐⭐⭐⭐⭐ SOLUSI 1: MINORITY CLASS OVERSAMPLING (TERBAIK!)

### Strategi: Generate Synthetic Minority Samples

**Implementation:** Modify `generate_ground_truth_crops.py`

#### **Step 1: Hitung Class Distribution**

Setelah split train/val/test, cek distribusi:

```python
# After splitting images (around line 800+)
print("\n[CLASS BALANCE] Analyzing class distribution...")

# Count samples per class in train set
train_class_counts = {}
for img_path in train_images:
    # Get class from annotation
    for ann in dataset_annotations:
        if ann['image_path'] == img_path:
            class_name = ann['class_name']
            train_class_counts[class_name] = train_class_counts.get(class_name, 0) + 1

print(f"[TRAIN DISTRIBUTION] {train_class_counts}")

# Calculate imbalance ratio
max_count = max(train_class_counts.values())
min_count = min(train_class_counts.values())
imbalance_ratio = max_count / min_count

print(f"[IMBALANCE] Ratio: {imbalance_ratio:.1f}:1 ({'SEVERE' if imbalance_ratio > 10 else 'MODERATE'})")
```

#### **Step 2: Apply SMOTE-like Oversampling**

**Buat fungsi baru di `generate_ground_truth_crops.py`:**

```python
def oversample_minority_classes(crops_by_class, target_ratio=0.5):
    """
    Oversample minority classes using aggressive augmentation

    Args:
        crops_by_class: Dict {class_name: [crop_paths]}
        target_ratio: Target minority/majority ratio (0.5 = 50% of majority)

    Returns:
        Augmented crops_by_class with synthetic minority samples
    """
    from PIL import Image, ImageEnhance, ImageFilter
    import random

    # Find majority and minority classes
    class_counts = {k: len(v) for k, v in crops_by_class.items()}
    max_count = max(class_counts.values())
    majority_class = max(class_counts, key=class_counts.get)

    print(f"\n[OVERSAMPLING] Target minority samples: {int(max_count * target_ratio)} each")
    print(f"[OVERSAMPLING] Majority class: {majority_class} ({max_count} samples)")

    augmented_crops = crops_by_class.copy()

    for class_name, crop_paths in crops_by_class.items():
        if class_name == majority_class:
            continue

        current_count = len(crop_paths)
        target_count = int(max_count * target_ratio)
        needed = target_count - current_count

        if needed <= 0:
            print(f"[SKIP] {class_name}: {current_count} samples (enough)")
            continue

        print(f"[AUGMENT] {class_name}: {current_count} → {target_count} (+{needed} synthetic)")

        # Generate synthetic samples through aggressive augmentation
        synthetic_crops = []
        for i in range(needed):
            # Randomly pick source image
            source_path = random.choice(crop_paths)
            img = Image.open(source_path).convert('RGB')

            # Apply random augmentation chain (3-5 transforms)
            num_transforms = random.randint(3, 5)
            for _ in range(num_transforms):
                transform_type = random.choice([
                    'brightness', 'contrast', 'sharpness', 'blur',
                    'hflip', 'vflip', 'rotate'
                ])

                if transform_type == 'brightness':
                    factor = random.uniform(0.7, 1.3)
                    img = ImageEnhance.Brightness(img).enhance(factor)
                elif transform_type == 'contrast':
                    factor = random.uniform(0.7, 1.3)
                    img = ImageEnhance.Contrast(img).enhance(factor)
                elif transform_type == 'sharpness':
                    factor = random.uniform(0.5, 2.5)
                    img = ImageEnhance.Sharpness(img).enhance(factor)
                elif transform_type == 'blur':
                    radius = random.uniform(0.5, 1.5)
                    img = img.filter(ImageFilter.GaussianBlur(radius))
                elif transform_type == 'hflip':
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif transform_type == 'vflip':
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                elif transform_type == 'rotate':
                    angle = random.choice([-15, -10, -5, 5, 10, 15])
                    img = img.rotate(angle, fillcolor=(255, 255, 255))

            # Save synthetic crop
            source_name = Path(source_path).stem
            synthetic_name = f"{source_name}_syn{i:03d}.jpg"
            synthetic_path = Path(source_path).parent / synthetic_name
            img.save(synthetic_path, quality=95)
            synthetic_crops.append(str(synthetic_path))

        # Add synthetic crops to class
        augmented_crops[class_name].extend(synthetic_crops)
        print(f"[DONE] {class_name}: Created {len(synthetic_crops)} synthetic samples")

    return augmented_crops
```

#### **Step 3: Apply After Train Split (Before Saving)**

Dalam `generate_ground_truth_crops.py`, setelah crop generation tapi sebelum selesai:

```python
# Around line 900+ (after all crops saved to train/val/test folders)

# MINORITY CLASS OVERSAMPLING for train set only
print("\n" + "="*60)
print("MINORITY CLASS OVERSAMPLING (Train Set Only)")
print("="*60)

train_crops_path = output_dir / "train"
crops_by_class = {}

# Collect crops by class
for class_folder in train_crops_path.iterdir():
    if class_folder.is_dir():
        class_name = class_folder.name
        crop_files = list(class_folder.glob("*.jpg"))
        crops_by_class[class_name] = [str(f) for f in crop_files]

# Check if oversampling is needed
class_counts = {k: len(v) for k, v in crops_by_class.items()}
max_count = max(class_counts.values())
min_count = min(class_counts.values())
imbalance_ratio = max_count / min_count

if imbalance_ratio > 5.0:  # Only if severe imbalance
    print(f"[SEVERE IMBALANCE] Ratio: {imbalance_ratio:.1f}:1 - Applying oversampling...")

    # Apply oversampling (target 50% of majority)
    augmented_crops = oversample_minority_classes(crops_by_class, target_ratio=0.5)

    # Summary
    print("\n[SUMMARY] Before → After:")
    for class_name in sorted(crops_by_class.keys()):
        before = len(crops_by_class[class_name])
        after = len(augmented_crops[class_name])
        print(f"  {class_name}: {before} → {after} (+{after-before})")
else:
    print(f"[OK] Imbalance ratio {imbalance_ratio:.1f}:1 is acceptable (< 5:1)")

print("="*60)
```

---

## ⭐⭐⭐⭐ SOLUSI 2: FOCAL LOSS PARAMETER TUNING

### Current: α=1.0, γ=1.5
### Optimized: α=0.25, γ=2.0 (lebih fokus ke minority)

**Cara:**

Edit `main_pipeline.py`, tambahkan argumen:

```python
parser.add_argument("--focal-alpha", type=float, default=0.25,
                   help="Focal loss alpha (0.25 for minority class focus)")
parser.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma (2.0 standard)")
```

Pass ke training script:

```python
# In classification training call
cls_cmd = [
    sys.executable,
    "scripts/training/12_train_pytorch_classification.py",
    "--model", model_name,
    "--data", str(crop_data_path),
    "--epochs", str(args.epochs_cls),
    "--batch", "64",
    "--lr", "0.0005",
    "--loss", "focal",
    "--focal_alpha", str(args.focal_alpha),
    "--focal_gamma", str(args.focal_gamma),
    # ...
]
```

**Test berbagai kombinasi:**

```bash
# Standard (current)
python main_pipeline.py --focal-alpha 1.0 --focal-gamma 1.5

# Paper recommended (minority focus)
python main_pipeline.py --focal-alpha 0.25 --focal-gamma 2.0

# Aggressive minority focus
python main_pipeline.py --focal-alpha 0.5 --focal-gamma 2.5
```

---

## ⭐⭐⭐ SOLUSI 3: CLASS-WEIGHTED SAMPLER (More Aggressive)

### Current: WeightedRandomSampler with inverse frequency
### Improved: Square root weighting (less aggressive, more stable)

**Edit `12_train_pytorch_classification.py` baris 192-217:**

```python
def create_weighted_sampler(dataset, strategy='inverse'):
    """Create weighted random sampler for balanced training

    Args:
        dataset: PyTorch dataset
        strategy: 'inverse' (default), 'sqrt' (milder), 'square' (aggressive)
    """
    # Count samples per class
    labels = []
    for _, label in dataset:
        labels.append(label)

    class_counts = Counter(labels)
    total_samples = len(labels)

    print(f"[CLASS DISTRIBUTION] {dict(class_counts)}")

    # Calculate sample weights based on strategy
    sample_weights = []
    for label in labels:
        count = class_counts[label]

        if strategy == 'inverse':
            # Standard inverse frequency
            weight = total_samples / (len(class_counts) * count)
        elif strategy == 'sqrt':
            # Milder: square root of inverse
            weight = np.sqrt(total_samples / (len(class_counts) * count))
        elif strategy == 'square':
            # Aggressive: square of inverse
            weight = (total_samples / (len(class_counts) * count)) ** 2

        sample_weights.append(weight)

    print(f"[SAMPLING] Strategy: {strategy}")
    print(f"[SAMPLING] Weight range: {min(sample_weights):.2f} - {max(sample_weights):.2f}")

    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler
```

**Test:**

```python
# In main() function, line 291
weighted_sampler = create_weighted_sampler(train_dataset, strategy='square')  # Try 'square' for aggressive
```

---

## 📊 EXPECTED IMPROVEMENTS

### Dengan Solusi 1 (Oversampling):

| Model | Current | Expected | Improvement |
|-------|---------|----------|-------------|
| ResNet50 | 88.51% | 92-94% | +4-6% |
| ResNet101 | 93.19% | 95-97% | +2-4% |
| EfficientNet-B0 | 98.30% | 98.5-99.2% | +0.5-1% |

**Why:** More minority samples = ResNet dapat belajar pattern minority class

### Dengan Solusi 2 (Focal α=0.25, γ=2.0):

| Model | Current | Expected | Improvement |
|-------|---------|----------|-------------|
| All models | Baseline | +1-2% | Better minority recall |

**Why:** Focal loss lebih fokus ke hard examples (minority class)

### Dengan Solusi 3 (Aggressive weighting):

| Model | Current | Expected | Improvement |
|-------|---------|----------|-------------|
| ResNet50 | 88.51% | 90-92% | +2-4% |
| ResNet101 | 93.19% | 94-96% | +1-3% |

**Why:** Minority class muncul lebih sering di training batches

---

## ✅ REKOMENDASI URUTAN

### **Week 1: Kombinasi Solusi 1 + 2**

**Hari 1-2:** Implement Oversampling (Solusi 1)
- Edit `generate_ground_truth_crops.py`
- Re-generate crops dengan oversampling
- Expected: +4-6% untuk ResNet

**Hari 3-4:** Tuning Focal Loss (Solusi 2)
- Test α=0.25, γ=2.0
- Expected: +1-2% semua model

**Hari 5:** Analyze results & pick best combination

### **Expected Final Results:**

- **ResNet50:** 88.51% → **93-95%** (+5-7%)
- **ResNet101:** 93.19% → **96-98%** (+3-5%)
- **EfficientNet-B0:** 98.30% → **99.0-99.5%** (+0.7-1.2%)

**Target:** Semua model >95%, dengan EfficientNet-B0 mendekati 99.5%!

---

## 🎯 NEXT STEP

Pilih salah satu untuk START:

**A. SOLUSI 1 (Oversampling)** - Paling efektif tapi perlu re-generate crops
**B. SOLUSI 2 (Focal Loss)** - Tercepat, tinggal re-train
**C. SOLUSI 3 (Aggressive Sampling)** - Quick test, 1 hari

Atau kombinasi A+B untuk hasil maksimal!

**Mau mulai yang mana?**
