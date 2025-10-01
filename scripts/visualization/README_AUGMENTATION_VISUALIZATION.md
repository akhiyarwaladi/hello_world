# Augmentation Visualization for Paper

Script untuk generate visualisasi augmentasi yang digunakan di pipeline untuk keperluan paper.

## 📁 Output Structure

```
augmentation_examples/
├── augmentation_comparison_{class}_{name}.png       # Figure untuk paper (compact)
├── augmentation_examples_{class}_{name}.png         # Full examples (10 augmentations)
└── individual_augmentations/                        # Individual images
    └── {class}/
        └── {parasite_name}/
            ├── 1_original.png
            ├── 2_horizontal_flip.png
            ├── 3_vertical_flip.png
            ├── 4_rotation_15deg.png
            ├── 5_rotation_minus15deg.png
            ├── 6_brightness_increase.png
            ├── 7_brightness_decrease.png
            ├── 8_contrast_increase.png
            ├── 9_saturation_increase.png
            └── 10_combined_augment.png
```

## 🚀 Usage

### 1. Generate Full Examples (All 10 Augmentations)

```bash
# Lifecycle dataset
python scripts/visualization/visualize_augmentation.py \
  --image data/ground_truth_crops_224/lifecycle/crops/test/gametocyte/PA171697_crop_000.jpg \
  --output augmentation_examples

# Species dataset
python scripts/visualization/visualize_augmentation.py \
  --image data/ground_truth_crops_224/species/crops/test/P_falciparum/img001_crop_000.jpg \
  --output augmentation_examples

# Stages dataset
python scripts/visualization/visualize_augmentation.py \
  --image data/ground_truth_crops_224/stages/crops/test/ring/img001_crop_000.jpg \
  --output augmentation_examples
```

### 2. Generate Comparison Figure Only (For Paper)

Untuk paper, gunakan `--comparison-only` untuk generate figure yang lebih compact:

```bash
python scripts/visualization/visualize_augmentation.py \
  --image data/ground_truth_crops_224/lifecycle/crops/test/gametocyte/PA171697_crop_000.jpg \
  --output paper_figures \
  --comparison-only
```

**Output:** 2×3 grid dengan 6 augmentasi paling penting (Original, Horizontal Flip, Rotation +15°, Rotation -15°, Brightness↑, Combined)

### 3. Batch Processing (Multiple Parasites)

Generate untuk satu parasite dari setiap class:

```bash
# Lifecycle - 4 classes
for class in ring gametocyte trophozoite schizont; do
  image=$(find data/ground_truth_crops_224/lifecycle/crops/test/$class -name "*.jpg" -type f | head -1)
  python scripts/visualization/visualize_augmentation.py \
    --image "$image" \
    --output paper_figures/lifecycle \
    --comparison-only
done

# Species - 4 classes
for class in P_falciparum P_vivax P_malariae P_ovale; do
  image=$(find data/ground_truth_crops_224/species/crops/test/$class -name "*.jpg" -type f | head -1)
  python scripts/visualization/visualize_augmentation.py \
    --image "$image" \
    --output paper_figures/species \
    --comparison-only
done

# Stages - 4 classes
for class in ring schizont trophozoite gametocyte; do
  image=$(find data/ground_truth_crops_224/stages/crops/test/$class -name "*.jpg" -type f | head -1)
  python scripts/visualization/visualize_augmentation.py \
    --image "$image" \
    --output paper_figures/stages \
    --comparison-only
done
```

## 📊 Augmentation Techniques Visualized

### 1. **Original** (Baseline)
Ground truth crop resized to 224×224

### 2. **Horizontal Flip** (p=0.5)
Mirror image horizontally

### 3. **Vertical Flip** (p=0.3)
Mirror image vertically

### 4. **Rotation +15°**
Clockwise rotation

### 5. **Rotation -15°**
Counter-clockwise rotation

### 6. **Brightness Increase**
Enhance brightness (+0.3)

### 7. **Brightness Decrease**
Reduce brightness (-0.2)

### 8. **Contrast Increase**
Enhance contrast (+0.3)

### 9. **Saturation Increase**
Enhance saturation (+0.3)

### 10. **Combined Augmentation**
Horizontal flip + rotation + color jitter

## 📝 Notes for Paper

### Caption Example:

> **Figure X: Data Augmentation Pipeline**
> Example of augmentation techniques applied to parasite crops during training.
> (A) Original ground truth crop (224×224 pixels).
> (B) Horizontal flip (p=0.5).
> (C) Rotation +15° and -15° (random ±15°).
> (D) Brightness adjustment (±0.1).
> (E) Combined augmentation with flip, rotation, and color jitter.
> All augmentations are applied randomly during training to improve model robustness.

### Method Section Text Example:

> *Data Augmentation:* Ground truth crops were subjected to real-time augmentation
> during training, including random horizontal flip (p=0.5), vertical flip (p=0.3),
> rotation (±15°), and color jitter (brightness, contrast, saturation ±0.1, hue ±0.05).
> Images were resized to 224×224 pixels and normalized using ImageNet statistics
> (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

## 🎯 Comparison with Ref1

| Aspect | Our Pipeline | Ref1 |
|--------|-------------|------|
| **Training** | Real-time random (infinite variations) | 14× pre-generated fixed |
| **Val/Test** | No augmentation | 7× pre-generated (robustness test) |
| **Approach** | On-the-fly | Pre-generated |
| **Storage** | Low (original only) | High (14× + 7×) |
| **Flexibility** | High (different each epoch) | Low (same each epoch) |

## 💡 Tips

1. **For paper figures:** Use `--comparison-only` untuk compact visualization
2. **For supplementary:** Generate full examples dengan semua 10 augmentations
3. **Resolution:** Default 224×224 (sesuai pipeline), bisa diubah dengan `--image-size`
4. **Quality:** Output dpi=300 untuk publication-ready figures

## ⚠️ Requirements

```bash
pip install torch torchvision matplotlib pillow numpy
```

## 🔧 Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'torch'`
**Solution:** Install PyTorch terlebih dahulu

**Issue:** Image tidak ditemukan
**Solution:** Pastikan path image benar dan file ada

**Issue:** Output folder permission denied
**Solution:** Ubah output directory ke folder yang punya write permission
