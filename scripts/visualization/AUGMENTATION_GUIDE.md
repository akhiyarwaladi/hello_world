# Augmentation Visualization Guide

## 📊 High-Quality Augmentation Figures for Papers/Reports

Generate publication-quality data augmentation visualizations with configurable resolution and comprehensive augmentation techniques.

---

## 🚀 Quick Start

### Basic Usage (4 samples, 384x384 resolution)

```bash
python scripts/visualization/generate_high_quality_augmentation_figure.py \
  --images \
    "results/optA_*/experiments/experiment_iml_lifecycle/crops_gt_crops/crops/test/gametocyte/PA171793_crop_001.jpg" \
    "results/optA_*/experiments/experiment_iml_lifecycle/crops_gt_crops/crops/test/ring/PA171690_crop_000.jpg" \
    "results/optA_*/experiments/experiment_iml_lifecycle/crops_gt_crops/crops/test/schizont/PA171819_crop_000.jpg" \
    "results/optA_*/experiments/experiment_iml_lifecycle/crops_gt_crops/crops/test/trophozoite/PA171771_crop_002.jpg" \
  --output "luaran/figures/augmentation_figure.png" \
  --size 384 \
  --dpi 300
```

### High-Resolution (Publication Quality - 512x512)

```bash
python scripts/visualization/generate_high_quality_augmentation_figure.py \
  --images image1.jpg image2.jpg image3.jpg image4.jpg \
  --output "paper_figure_augmentation.png" \
  --size 512 \
  --dpi 300
```

### Ultra High-Resolution (Poster/Presentation - 1024x1024)

```bash
python scripts/visualization/generate_high_quality_augmentation_figure.py \
  --images image1.jpg image2.jpg \
  --output "poster_augmentation.png" \
  --size 1024 \
  --dpi 300
```

---

## 📋 Augmentation Techniques Included

The script generates **14 augmentations** per sample:

### Row 1 (7 augmentations):
1. **Original** - Baseline image
2. **90° clockwise** - Rotation -90°
3. **180° clockwise** - Rotation 180°
4. **90° anti-clockwise** - Rotation 90°
5. **270° clockwise** - Rotation -270°
6. **Brightness 0.8** - Darker (×0.8)
7. **Brightness 1.2** - Brighter (×1.2)

### Row 2 (7 augmentations):
8. **Contrast 0.5** - Lower contrast (×0.5)
9. **Brightness 1.5** - Much brighter (×1.5)
10. **Flip horizontal** - Mirror horizontally
11. **Flip vertical** - Mirror vertically
12. **Saturation 0.5** - Less saturated (×0.5)
13. **Saturation 1.5** - More saturated (×1.5)
14. **Sharpness 1.5** - Sharper (×1.5)

---

## 🎨 Quality Comparison

| Resolution | File Size | Use Case |
|------------|-----------|----------|
| **224×224** | ~1 MB | Training (low quality) ❌ |
| **384×384** | ~7-8 MB | Reports, Theses ✅ |
| **512×512** | ~12-15 MB | Papers, Journals ✅✅ |
| **1024×1024** | ~40-50 MB | Posters, Presentations ⭐ |

---

## 📖 Parameters

```bash
--images          Path(s) to parasite crop images (1-4 recommended)
--output          Output file path (PNG format)
--size            Image size per cell (default: 512)
--dpi             Output DPI (default: 300 for publication)
```

---

## 💡 Tips

### 1. Choose Diverse Samples
Select images representing different:
- **Classes**: gametocyte, ring, schizont, trophozoite
- **Quality**: Clear, well-defined parasites
- **Sizes**: Mix of small and large parasites

### 2. Resolution Selection
- **Paper submission**: 512×512, 300 DPI
- **Thesis/Report**: 384×384, 300 DPI
- **Presentation**: 1024×1024, 300 DPI
- **Quick preview**: 256×256, 150 DPI

### 3. Finding Good Samples

```bash
# List all test crops by class
ls results/optA_*/experiments/experiment_iml_lifecycle/crops_gt_crops/crops/test/*/

# Preview a crop (Windows)
start results/optA_*/experiments/experiment_iml_lifecycle/crops_gt_crops/crops/test/gametocyte/PA171793_crop_001.jpg
```

---

## 🔍 Output Structure

The generated figure follows publication standards:

```
┌─────────────────────────────────────────────────────────────────┐
│  Figure 1. Example of data augmentation on the detected         │
│  infected cell conducted on the training dataset                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GAMETOCYTE │ [Original] [90°] [180°] [90° anti] [270°] [B0.8] [B1.2] │
│             │ [C0.5] [B1.5] [FlipH] [FlipV] [S0.5] [S1.5] [Sharp]    │
│                                                                 │
│  RING       │ [Original] [90°] [180°] [90° anti] [270°] [B0.8] [B1.2] │
│             │ [C0.5] [B1.5] [FlipH] [FlipV] [S0.5] [S1.5] [Sharp]    │
│                                                                 │
│  SCHIZONT   │ [Original] [90°] [180°] [90° anti] [270°] [B0.8] [B1.2] │
│             │ [C0.5] [B1.5] [FlipH] [FlipV] [S0.5] [S1.5] [Sharp]    │
│                                                                 │
│  TROPHOZOITE│ [Original] [90°] [180°] [90° anti] [270°] [B0.8] [B1.2] │
│             │ [C0.5] [B1.5] [FlipH] [FlipV] [S0.5] [S1.5] [Sharp]    │
└─────────────────────────────────────────────────────────────────┘
```

**Format**: 7 columns × (2 rows per sample) × n_samples

---

## ⚡ Performance

| Samples | Size | Generation Time |
|---------|------|-----------------|
| 1 image | 512×512 | ~2-3 seconds |
| 2 images | 512×512 | ~4-5 seconds |
| 4 images | 512×512 | ~8-10 seconds |
| 4 images | 1024×1024 | ~15-20 seconds |

---

## 🎯 Example Outputs

### Successfully Generated:
- `luaran/figures/augmentation_visualization_high_quality.png`
  - Resolution: 384×384 per cell
  - DPI: 300
  - File size: 7.63 MB
  - Samples: 4 (all classes)

---

## 🐛 Troubleshooting

### Issue: File size too large
**Solution**: Reduce `--size` or `--dpi`
```bash
--size 256 --dpi 150  # Smaller file (~2-3 MB)
```

### Issue: Images look pixelated
**Solution**: Increase `--size`
```bash
--size 1024 --dpi 300  # Ultra high quality
```

### Issue: Script too slow
**Solution**: Reduce number of samples or resolution
```bash
--images image1.jpg image2.jpg  # Only 2 samples
--size 384  # Faster than 1024
```

---

## 📚 Related Scripts

- `visualize_augmentation.py` - Old script (224×224, lower quality)
- `generate_improved_gradcam.py` - Grad-CAM visualizations
- `generate_detection_classification_figures.py` - Detection + classification outputs

---

## 📝 Notes

1. **White Background**: Rotations use white fill (no black corners) for medical images
2. **BICUBIC Interpolation**: High-quality resampling for better image quality
3. **Publication Ready**: 300 DPI meets most journal requirements
4. **Flexible**: Works with any parasite crop images

---

**Last Updated**: 2025-10-08
**Script**: `scripts/visualization/generate_high_quality_augmentation_figure.py`
