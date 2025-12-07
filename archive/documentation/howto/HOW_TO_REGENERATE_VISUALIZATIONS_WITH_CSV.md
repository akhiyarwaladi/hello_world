# 🎨 How to Regenerate Visualizations WITH CSV Metadata

**Purpose:** Re-generate visualizations dengan CSV metadata untuk membantu pilih gambar terbaik untuk paper

**Date:** 2025-10-26

---

## 🎯 WHY Use This?

**Problem Lama:**
- Visualizations hanya generate gambar PNG ❌
- Tidak ada info mana gambar yang bagus untuk paper ❌
- Harus manual cek satu-satu 100+ gambar ❌
- Tidak tahu mana yang correct/incorrect predictions ❌

**Solution Baru:**
- ✅ Generate gambar PNG **+ CSV metadata**
- ✅ CSV berisi: correct/incorrect, confidence, ground truth vs predicted
- ✅ Ranking otomatis: "Best for paper", "Good for paper", "Challenging case"
- ✅ Easy sorting: sort by 'paper_score' untuk find best images instantly!

---

## 📊 CSV Output Format

### Detection CSV (`detection_metadata.csv`)

| Column | Description | Example |
|--------|-------------|---------|
| `image_name` | Image filename (without extension) | `PA171697` |
| `image_file` | Full path to generated PNG | `visualizations/pred_detection_yolo11/PA171697.png` |
| `n_gt_boxes` | Number of ground truth boxes | `3` |
| `n_pred_boxes` | Number of predicted boxes | `3` |
| `n_correct_matches` | Number of correct detections (IoU ≥0.5) | `3` |
| `n_false_positives` | Number of false positive detections | `0` |
| `n_false_negatives` | Number of missed detections | `0` |
| `avg_confidence` | Average confidence of predictions | `0.876` |
| `status` | Detection status | `Perfect` |
| `paper_score` | Paper suitability score (0-10) | `10` |
| `recommendation` | Recommendation | `Best for paper` |

**Status Values:**
- `Perfect` - All boxes detected correctly (n_gt == n_pred == n_correct)
- `Partial correct` - Some correct, no FP/FN
- `Missing detections (FN)` - Missed some boxes
- `False positives (FP)` - Extra wrong boxes
- `Mixed (FP + FN)` - Both missed and extra boxes
- `Empty (no GT, no predictions)` - No objects

**Paper Score:**
- **10** = Perfect detection + high confidence (>0.8) → **Best for paper!**
- **9** = Perfect detection + medium confidence
- **7-8** = Good detection (some correct, no FP)
- **5-6** = Challenging case (mixed FP+FN) → **Could show limitations**
- **3-4** = Poor detection
- **2** = Not ideal

### Classification CSV

**Image-level:** `classification_metadata_images.csv`

| Column | Description | Example |
|--------|-------------|---------|
| `image_name` | Image filename | `PA171697` |
| `n_boxes` | Total boxes classified | `3` |
| `n_correct` | Correctly classified boxes | `3` |
| `n_incorrect` | Incorrectly classified boxes | `0` |
| `accuracy` | Accuracy for this image | `1.000` |
| `avg_confidence` | Average confidence | `0.912` |
| `status` | Classification status | `All correct` |
| `paper_score` | Paper suitability score (0-10) | `10` |
| `recommendation` | Recommendation | `Best for paper` |

**Box-level:** `classification_metadata_boxes.csv`

| Column | Description | Example |
|--------|-------------|---------|
| `image_name` | Image filename | `PA171697` |
| `crop_idx` | Box index in image | `0` |
| `gt_class` | Ground truth class | `schizont` |
| `pred_class` | Predicted class | `schizont` |
| `confidence` | Prediction confidence | `0.943` |
| `correct` | Correct prediction? | `Yes` |

**Status Values:**
- `All correct` - Every box classified correctly
- `Mixed (some correct, some wrong)` - Mix of correct & errors
- `All wrong` - Every box misclassified
- `No boxes` - No objects to classify

---

## 🚀 METHOD 1: Re-run from Main Pipeline (RECOMMENDED)

**Use Case:** Re-generate ALL visualizations untuk existing experiment

### Step 1: Identify Your Experiment

```bash
# List available experiments
ls results/
```

Output:
```
optA_20251016_200330/
optA_20251018_154532/
... etc
```

### Step 2: Re-run Analysis Stage ONLY

```bash
# For MD_2019 dataset
python main_pipeline.py \
  --continue-from optA_20251016_200330 \
  --dataset md_2019_stages \
  --start-stage analysis
```

**What Happens:**
1. ✅ Skips detection (already trained)
2. ✅ Skips crop generation (already done)
3. ✅ Skips classification (already trained)
4. ✅ **Re-runs analysis stage** (includes visualization generation)
5. ✅ Generates NEW visualizations + CSV metadata
6. ✅ **Overwrites** old visualizations (SAFE - no training data affected!)

**Important Notes:**
- `--start-stage analysis` = Start from analysis, skip all training
- This is **SAFE** - it ONLY regenerates analysis files, doesn't touch trained models
- Old visualizations will be replaced with new ones (with CSV!)

### Step 3: Check Output

```bash
cd results/optA_20251016_200330/experiments/experiment_md_2019_stages/visualizations
```

You'll see:
```
visualizations/
├── pred_detection_yolo10/
│   ├── PA171697.png
│   ├── PA171698.png
│   ├── ... (images)
│   └── detection_metadata.csv  ← NEW! 📊
│
├── pred_detection_yolo11/
│   ├── ... (images)
│   └── detection_metadata.csv  ← NEW! 📊
│
├── pred_classification_efficientnet_b0_focal/
│   ├── ... (images)
│   ├── classification_metadata_images.csv  ← NEW! 📊
│   └── classification_metadata_boxes.csv   ← NEW! 📊
│
└── ... (other models)
```

---

## 🎯 METHOD 2: Standalone Script (QUICK TEST)

**Use Case:** Test on one experiment quickly without running full pipeline

### Command

```bash
python scripts/pipeline/generate_visualizations_with_metadata.py \
  --experiment-dir "results/optA_20251016_200330/experiments/experiment_md_2019_stages" \
  --dataset-name md_2019_stages \
  --num-classes 3 \
  --max-images 10
```

**Parameters:**
- `--experiment-dir`: Path to experiment folder
- `--dataset-name`: Dataset name (md_2019_stages, iml_lifecycle, mp_idb_species, mp_idb_stages)
- `--num-classes`: Number of classes (3 for MD_2019, 4 for others)
- `--max-images`: Limit images (optional, use for quick test)

**Number of Classes by Dataset:**
- `md_2019_stages`: 3 classes (ring, schizont, trophozoite)
- `iml_lifecycle`: 4 classes (ring, trophozoite, schizont, gametocyte)
- `mp_idb_species`: 4 classes (P_falciparum, P_vivax, P_malariae, P_ovale)
- `mp_idb_stages`: 4 classes (ring, trophozoite, schizont, gametocyte)

### Quick Test (10 images only)

```bash
# Generate only first 10 images to test
python scripts/pipeline/generate_visualizations_with_metadata.py \
  --experiment-dir "results/optA_20251016_200330/experiments/experiment_md_2019_stages" \
  --dataset-name md_2019_stages \
  --num-classes 3 \
  --max-images 10
```

### Full Generation (ALL images)

```bash
# Generate ALL images (remove --max-images)
python scripts/pipeline/generate_visualizations_with_metadata.py \
  --experiment-dir "results/optA_20251016_200330/experiments/experiment_md_2019_stages" \
  --dataset-name md_2019_stages \
  --num-classes 3
```

---

## 📖 HOW TO USE CSV FOR PAPER SELECTION

### Example Workflow: Find Best Detection Images

1. **Open CSV in Excel/Pandas**
   ```python
   import pandas as pd
   df = pd.read_csv('visualizations/pred_detection_yolo11/detection_metadata.csv')
   ```

2. **Sort by paper_score (descending)**
   ```python
   df_sorted = df.sort_values('paper_score', ascending=False)
   ```

3. **Filter best images**
   ```python
   # Best for paper (score ≥ 9)
   best_images = df_sorted[df_sorted['paper_score'] >= 9]
   print(best_images[['image_name', 'status', 'avg_confidence', 'recommendation']])
   ```

4. **Get image paths**
   ```python
   # Top 5 best images
   for idx, row in best_images.head(5).iterrows():
       print(f"{row['image_name']}: {row['image_file']}")
   ```

5. **Copy to paper folder**
   ```bash
   # Copy selected images
   cp visualizations/pred_detection_yolo11/PA171697.png luaran/auto_generated/figures/
   ```

### Example: Find Challenging Cases

Useful untuk show model limitations in paper:

```python
# Find images with mixed FP + FN (challenging)
challenging = df[df['status'] == 'Mixed (FP + FN)']

# Or find images with high confidence BUT errors
high_conf_errors = df[(df['avg_confidence'] > 0.8) & (df['n_false_positives'] > 0)]
```

### Example: Classification Error Analysis

```python
# Load box-level data
boxes = pd.read_csv('visualizations/pred_classification_efficientnet_b0_focal/classification_metadata_boxes.csv')

# Find misclassified boxes
errors = boxes[boxes['correct'] == 'No']

# Group by predicted class to see confusion patterns
confusion = errors.groupby(['gt_class', 'pred_class']).size()
print("Confusion patterns:")
print(confusion)

# Example output:
# gt_class    pred_class
# ring        trophozoite    5
# schizont    ring          3
# trophozoite ring          8
```

---

## 🔄 Re-run for Multiple Datasets

If you have multi-dataset experiment:

```bash
# For each dataset
for dataset in iml_lifecycle mp_idb_species mp_idb_stages md_2019_stages
do
  echo "Processing $dataset..."
  python main_pipeline.py \
    --continue-from optA_20251016_200330 \
    --dataset $dataset \
    --start-stage analysis
done
```

---

## ⚠️ IMPORTANT NOTES

### Will This Break My Experiment?

**NO!** Re-running analysis stage is **100% SAFE**:
- ✅ Detection models: NOT touched
- ✅ Classification models: NOT touched
- ✅ Crops: NOT regenerated
- ✅ Training data: NOT modified
- ✅ Experiment metadata: NOT changed

**What WILL be replaced:**
- ⚠️ Old visualization PNG files → Replaced with new ones
- ⚠️ Old analysis JSON files → Regenerated (but same content)
- ✅ **NEW:** CSV metadata files added

### Disk Space

Each visualization folder contains:
- ~100-300 PNG images (~50-150 MB)
- 1-2 CSV files (~100-500 KB)
- **Total:** ~50-150 MB per model per dataset

For full experiment (3 detection + 6 classification models):
- ~450-1350 MB for visualizations
- Negligible (~5 MB) for CSVs

### Time Estimate

**Detection Visualizations:**
- ~0.5-1 second per image
- 100 images × 3 models = **~5-10 minutes**

**Classification Visualizations:**
- ~1-2 seconds per image (includes model inference)
- 100 images × 6 models = **~10-20 minutes**

**Total for 1 dataset:** ~15-30 minutes

---

## 🎓 EXAMPLE: Complete Workflow

```bash
# 1. Re-generate visualizations with CSV for MD_2019
python main_pipeline.py \
  --continue-from optA_20251016_200330 \
  --dataset md_2019_stages \
  --start-stage analysis

# 2. Navigate to output
cd results/optA_20251016_200330/experiments/experiment_md_2019_stages/visualizations

# 3. Open CSV
python
>>> import pandas as pd
>>> det = pd.read_csv('pred_detection_yolo11/detection_metadata.csv')
>>> det_sorted = det.sort_values('paper_score', ascending=False)

# 4. Show top 5 best images
>>> print(det_sorted.head(5)[['image_name', 'status', 'paper_score']])

# 5. Show challenging cases
>>> challenging = det[det['paper_score'] == 5]
>>> print(challenging[['image_name', 'status', 'n_false_positives', 'n_false_negatives']])

# 6. Classification analysis
>>> cls_img = pd.read_csv('pred_classification_efficientnet_b0_focal/classification_metadata_images.csv')
>>> cls_img_sorted = cls_img.sort_values('paper_score', ascending=False)
>>> print(cls_img_sorted.head(5)[['image_name', 'accuracy', 'paper_score']])

# 7. Box-level errors
>>> cls_box = pd.read_csv('pred_classification_efficientnet_b0_focal/classification_metadata_boxes.csv')
>>> errors = cls_box[cls_box['correct'] == 'No']
>>> print(errors[['image_name', 'gt_class', 'pred_class', 'confidence']])
```

---

## 📝 SUMMARY

**3 Scripts Created:**
1. `scripts/visualization/generate_detection_only_with_metadata.py` - Detection + CSV
2. `scripts/visualization/generate_classification_only_with_metadata.py` - Classification + CSV
3. `scripts/pipeline/generate_visualizations_with_metadata.py` - Wrapper for both

**2 Methods to Run:**
1. **Main Pipeline** (recommended): `python main_pipeline.py --continue-from <exp> --dataset <dataset> --start-stage analysis`
2. **Standalone** (quick test): `python scripts/pipeline/generate_visualizations_with_metadata.py --experiment-dir <path> --dataset-name <name> --num-classes <n>`

**CSV Output:**
- `detection_metadata.csv` - Image-level detection stats
- `classification_metadata_images.csv` - Image-level classification stats
- `classification_metadata_boxes.csv` - Box-level classification details

**Key Benefits:**
- ✅ Instant ranking by paper suitability
- ✅ Find perfect detections automatically
- ✅ Identify challenging/failure cases
- ✅ Analyze confusion patterns
- ✅ No manual checking of 100s of images!

---

**Last Updated:** 2025-10-26
**Status:** ✅ Ready to use
**Next Step:** Run visualization generation and analyze CSV files! 🚀
