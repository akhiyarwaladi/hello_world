# 📊 How to Generate Augmentation Figures for Paper

## ✅ Script Sudah Dibuat

Script: `scripts/visualization/visualize_augmentation.py`

## 🎯 Tujuan

Generate visualisasi augmentasi yang digunakan di pipeline untuk paper, menampilkan:
- Original crop (224×224)
- Berbagai teknik augmentasi yang diterapkan
- Format yang siap untuk paper

## 📋 Langkah-Langkah

### 1. Pastikan Environment Aktif

```bash
# Aktifkan conda/venv environment yang ada PyTorch
conda activate your_env
# atau
source venv/bin/activate
```

### 2. Generate Figure untuk Paper (Recommended)

**Untuk 1 parasite (compact figure untuk paper):**

```bash
python scripts/visualization/visualize_augmentation.py \
  --image data/ground_truth_crops_224/lifecycle/crops/test/gametocyte/PA171697_crop_000.jpg \
  --output paper_figures \
  --comparison-only
```

**Output:** `paper_figures/augmentation_comparison_gametocyte_PA171697_crop_000.png`
- Format: 2 rows × 3 cols (6 augmentations)
- Resolution: 300 dpi (publication-ready)
- Size: ~12×8 inches

### 3. Generate untuk Semua Classes (Lifecycle)

```bash
# Windows (Git Bash)
for class in ring gametocyte trophozoite schizont; do
  image=$(find data/ground_truth_crops_224/lifecycle/crops/test/$class -name "*.jpg" -type f | head -1)
  python scripts/visualization/visualize_augmentation.py \
    --image "$image" \
    --output paper_figures/lifecycle \
    --comparison-only
done

# Linux/Mac
for class in ring gametocyte trophozoite schizont; do
  image=$(find data/ground_truth_crops_224/lifecycle/crops/test/$class -name "*.jpg" -type f | head -1)
  python scripts/visualization/visualize_augmentation.py \
    --image "$image" \
    --output paper_figures/lifecycle \
    --comparison-only
done
```

### 4. Generate Full Examples (10 Augmentations)

Untuk supplementary materials atau detailed analysis:

```bash
python scripts/visualization/visualize_augmentation.py \
  --image data/ground_truth_crops_224/lifecycle/crops/test/gametocyte/PA171697_crop_000.jpg \
  --output augmentation_examples
```

**Output:**
- `augmentation_examples/augmentation_examples_gametocyte_PA171697_crop_000.png` (grid 10 augmentations)
- `augmentation_examples/augmentation_comparison_gametocyte_PA171697_crop_000.png` (compact)
- `augmentation_examples/individual_augmentations/gametocyte/PA171697_crop_000/*.png` (individual files)

## 📊 Output Structure

### Comparison Figure (For Paper)

```
┌─────────────┬─────────────┬─────────────┐
│  Original   │   H-Flip    │  Rotation   │
│  (Red box)  │             │    +15°     │
├─────────────┼─────────────┼─────────────┤
│  Rotation   │ Brightness↑ │  Combined   │
│    -15°     │             │  Augment    │
└─────────────┴─────────────┴─────────────┘

Title: "Data Augmentation Pipeline - {Class Name}"
Size: 12×8 inches, 300 dpi
```

### Full Examples Grid

```
┌──────┬──────┬──────┬──────┬──────┐
│  1   │  2   │  3   │  4   │  5   │
├──────┼──────┼──────┼──────┼──────┤
│  6   │  7   │  8   │  9   │  10  │
└──────┴──────┴──────┴──────┴──────┘

1. Original
2. Horizontal Flip
3. Vertical Flip
4. Rotation +15°
5. Rotation -15°
6. Brightness↑
7. Brightness↓
8. Contrast↑
9. Saturation↑
10. Combined
```

## 📝 Untuk Paper

### Figure Caption

```
Figure X: Data augmentation techniques applied to parasite classification.
(A) Original ground truth crop (224×224 pixels). (B) Horizontal flip (p=0.5).
(C) Rotation +15° (random ±15°). (D) Rotation -15°. (E) Brightness increase.
(F) Combined augmentation (flip + rotation + color jitter). All augmentations
are applied randomly during training to improve model generalization and
robustness to imaging variations.
```

### Method Section

```
Data Augmentation: During training, ground truth crops were subjected to
real-time augmentation including random horizontal flip (p=0.5), vertical
flip (p=0.3), random rotation (±15°), and color jitter (brightness, contrast,
saturation ±0.1, hue ±0.05). All images were resized to 224×224 pixels and
normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]). Unlike traditional pre-generated augmentation,
our pipeline employs on-the-fly random augmentation, exposing the model to
infinite variations across epochs, thereby enhancing generalization capability.
```

## 🎯 Augmentasi yang Divisualisasikan

| No | Augmentation | Parameters | Purpose |
|----|-------------|------------|---------|
| 1 | Original | - | Baseline |
| 2 | Horizontal Flip | p=1.0 (forced) | Orientation invariance |
| 3 | Vertical Flip | p=1.0 (forced) | Orientation invariance |
| 4 | Rotation +15° | Fixed 15° | Handle rotated samples |
| 5 | Rotation -15° | Fixed -15° | Handle rotated samples |
| 6 | Brightness↑ | +0.3 | Handle bright microscopy |
| 7 | Brightness↓ | -0.2 | Handle dim microscopy |
| 8 | Contrast↑ | +0.3 | Enhance features |
| 9 | Saturation↑ | +0.3 | Color variation |
| 10 | Combined | Flip+Rot+Color | Real-world scenario |

## ⚡ Quick Commands

### Generate 1 figure untuk paper (fastest):

```bash
# Pilih 1 parasite image yang bagus
python scripts/visualization/visualize_augmentation.py \
  --image data/ground_truth_crops_224/lifecycle/crops/test/gametocyte/PA171697_crop_000.jpg \
  --output paper_figures \
  --comparison-only
```

### Generate untuk semua datasets:

```bash
# Script batch (save as generate_all_figures.sh)
#!/bin/bash

datasets=("lifecycle" "species" "stages")

for dataset in "${datasets[@]}"; do
  echo "Processing $dataset..."

  # Find first test image from each class
  for class_dir in data/ground_truth_crops_224/$dataset/crops/test/*/; do
    class_name=$(basename "$class_dir")
    image=$(find "$class_dir" -name "*.jpg" -type f | head -1)

    if [ -n "$image" ]; then
      echo "  Generating for $class_name..."
      python scripts/visualization/visualize_augmentation.py \
        --image "$image" \
        --output "paper_figures/$dataset" \
        --comparison-only
    fi
  done
done

echo "All figures generated in paper_figures/"
```

```bash
chmod +x generate_all_figures.sh
./generate_all_figures.sh
```

## 🔧 Troubleshooting

### Error: ModuleNotFoundError: No module named 'torch'

**Solusi:**
```bash
# Install PyTorch
pip install torch torchvision

# Atau jika pakai conda
conda install pytorch torchvision -c pytorch
```

### Error: Image not found

**Solusi:**
```bash
# Cek apakah ground truth crops sudah di-generate
ls data/ground_truth_crops_224/

# Jika belum, run pipeline dulu untuk generate crops
python run_multiple_models_pipeline_OPTION_A.py \
  --dataset iml_lifecycle \
  --include yolo11 \
  --classification-models densenet121 \
  --epochs-det 1 \
  --epochs-cls 1 \
  --stop-stage crop
```

### Output tidak muncul

**Solusi:**
```bash
# Pastikan output directory bisa di-write
mkdir -p paper_figures
chmod 755 paper_figures

# Atau ganti ke directory lain
python scripts/visualization/visualize_augmentation.py \
  --image "path/to/image.jpg" \
  --output ~/Documents/figures
```

## 📦 Requirements

```bash
pip install torch torchvision matplotlib pillow numpy
```

## 💡 Tips

1. **Pilih parasite yang jelas dan representatif** untuk paper figure
2. **Gunakan `--comparison-only`** untuk paper (lebih compact)
3. **Resolution default 300 dpi** sudah publication-ready
4. **Simpan ke folder terpisah** (`paper_figures/`) untuk kemudahan organize
5. **Generate untuk semua classes** untuk show comprehensive augmentation strategy

## ✅ Checklist untuk Paper

- [ ] Generate comparison figure untuk minimal 1 parasite
- [ ] Pilih parasite yang jelas dan fokus
- [ ] Save dengan resolution tinggi (300 dpi)
- [ ] Tambahkan figure caption yang jelas
- [ ] Explain dalam method section
- [ ] Optional: Generate untuk semua classes (supplementary)

---

**Created:** 2025-10-01
**Script:** `scripts/visualization/visualize_augmentation.py`
**Documentation:** `scripts/visualization/README_AUGMENTATION_VISUALIZATION.md`
