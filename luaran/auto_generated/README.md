# Auto-Generated Publication Outputs

## ⚠️ CRITICAL WARNING

**DO NOT EDIT FILES IN THIS DIRECTORY MANUALLY**

All files in this directory are **automatically generated** from pipeline experiments. Any manual edits will be **permanently lost** when outputs are regenerated.

**If you need to make changes**:
1. Modify pipeline parameters or source code
2. Re-run experiments
3. Regenerate outputs using scripts below

---

## 📊 Contents Overview

This directory contains **44 auto-generated files** organized into 3 categories:

### 📈 Figures (31 files)
```
figures/
├── pipeline_diagrams/     (5 files)  - Architecture & flow diagrams
├── augmentation/          (21 files) - Data augmentation visualizations
└── performance/           (8 files)  - Performance analysis plots
```

### 📋 Tables (12 files)
```
tables/
├── classification/        (4 files)  - Classification metrics & results
├── detection/             (4 files)  - Detection performance & IoU analysis
└── statistics/            (4 files)  - Dataset statistics & distributions
```

### 🔍 Metadata (1 file)
```
_metadata.json             (1 file)   - Generation metadata & integrity tracking
```

---

## 🚀 One-Command Regeneration

### Regenerate Everything (Recommended)

```bash
# From project root
python scripts/publication/generate_all_publication_outputs.py

# Output:
# ✅ Generating 31 figures...
# ✅ Exporting 12 tables...
# ✅ Creating metadata.json...
# ✅ Verifying data integrity...
#
# Generated in ~5 minutes:
# - 31 publication-quality figures
# - 12 comprehensive data tables
# - 1 metadata file with integrity tracking
```

**What it does**:
1. Scans latest experiment results from `results/optA_*/`
2. Generates all figures (pipeline diagrams + augmentation + performance)
3. Exports all tables (classification + detection + statistics)
4. Creates `_metadata.json` with generation timestamps and integrity checksums
5. Verifies all files are present and valid

**Duration**: ~5 minutes (depends on number of experiments)

---

## 🔧 Selective Regeneration

### Future Feature (Not Yet Implemented)

The following flags are **planned but not yet available**. Currently, use the main regeneration command only.

```bash
# FUTURE: Selective regeneration by category
# python scripts/publication/generate_all_publication_outputs.py --figures-only
# python scripts/publication/generate_all_publication_outputs.py --tables-only

# FUTURE: Specific figure/table categories
# python scripts/publication/generate_all_publication_outputs.py --pipeline-diagrams
# python scripts/publication/generate_all_publication_outputs.py --classification-tables

# FUTURE: Specific experiment selection
# python scripts/publication/generate_all_publication_outputs.py \
#   --experiment-id optA_20251012_103000
```

**Current Usage**: Use the full regeneration command:
```bash
python scripts/publication/generate_all_publication_outputs.py
```

---

## 📁 Detailed File Catalog

### 1. figures/pipeline_diagrams/ (5 files)

**Purpose**: Visual architecture and workflow diagrams for papers/reports

| File | Description | Use In |
|------|-------------|--------|
| `pipeline_architecture_publication.png` | Main system architecture (publication quality) | All papers, Figure 1 |
| `detection_classification_flow.png` | Detection → Classification workflow | Methods section |
| `shared_classification_architecture.png` | Shared classification design | Architecture section |
| `yolo_comparison_diagram.png` | YOLO model comparison | Results section |
| `training_pipeline_flow.png` | Training pipeline flowchart | Methods section |
| `data_flow_diagram.png` | Data flow and transformations | System overview |

**Regeneration**:
```bash
# Full regeneration (includes all pipeline diagrams)
python scripts/publication/generate_all_publication_outputs.py
```

**Source**: Generated from pipeline architecture and configuration

---

### 2. figures/augmentation/ (21 files)

**Purpose**: Data augmentation visualizations for 3 datasets × 2 types (detection/classification)

**Per Dataset** (6 files each):
- `augmentation_[dataset]_detection.png` - Detection augmentation
- `augmentation_[dataset]_classification.png` - Classification augmentation
- `augmentation_[dataset]_detection_hr.png` - High-resolution variant
- `augmentation_[dataset]_classification_hr.png` - High-resolution variant
- `augmentation_[dataset]_combined.png` - Combined overview
- `augmentation_[dataset]_comparison.png` - Before/after comparison

**Datasets**:
1. `iml_lifecycle` (6 files)
2. `mp_idb_species` (6 files)
3. `mp_idb_stages` (6 files)

**Additional**:
- `augmentation_combined_all_datasets.png` - All datasets combined

**Regeneration**:
```bash
# Full regeneration (includes all augmentation figures)
python scripts/publication/generate_all_publication_outputs.py
```

**Source**: Generated from training data and augmentation parameters

---

### 3. figures/performance/ (8 files)

**Purpose**: Model performance visualizations and interpretability analysis

| File | Description | Use In |
|------|-------------|--------|
| `confusion_matrix_yolo11.png` | YOLO11 confusion matrix | Detection results |
| `confusion_matrix_densenet121.png` | DenseNet121 confusion matrix | Classification results |
| `confusion_matrix_efficientnet_b1.png` | EfficientNet-B1 confusion matrix | Classification results |
| `roc_curves_classification.png` | ROC curves (all classes) | Performance analysis |
| `pr_curves_detection.png` | Precision-Recall curves | Detection performance |
| `gradcam_visualization.png` | GradCAM interpretability | Model interpretability |
| `loss_curves_comparison.png` | Training loss comparison | Training analysis |
| `metrics_evolution.png` | Metrics evolution over epochs | Training dynamics |

**Regeneration**:
```bash
# Full regeneration (includes all performance figures)
python scripts/publication/generate_all_publication_outputs.py
```

**Source**: Generated from experiment results and trained models

---

### 4. tables/classification/ (4 files)

**Purpose**: Classification performance metrics and comparisons

| File | Description | Format | Use In |
|------|-------------|--------|--------|
| `table9_focal_loss.csv` | Table 9: Focal Loss results | CSV | Results section |
| `table9_class_balanced.csv` | Table 9: Class-Balanced Loss results | CSV | Results section |
| `classification_performance_summary.csv` | Overall classification metrics | CSV | Summary table |
| `classification_per_class_metrics.csv` | Per-class precision/recall/F1 | CSV | Detailed analysis |

**Example: table9_focal_loss.csv**
```csv
Dataset,Model,Accuracy,Balanced_Acc,Precision,Recall,F1_Score
iml_lifecycle,densenet121,0.6629,0.6245,0.6831,0.6629,0.6485
iml_lifecycle,efficientnet_b0,0.7191,0.6893,0.7342,0.7191,0.7124
iml_lifecycle,efficientnet_b1,0.8090,0.7845,0.8234,0.8090,0.8012
...
```

**Regeneration**:
```bash
# Full regeneration (includes all classification tables)
python scripts/publication/generate_all_publication_outputs.py
```

**Source**: Extracted from `results/optA_*/experiments/experiment_*/analysis_classification_*/`

---

### 5. tables/detection/ (4 files)

**Purpose**: Detection performance metrics and IoU analysis

| File | Description | Format | Use In |
|------|-------------|--------|--------|
| `detection_performance_all_datasets.csv` | YOLO comparison (all datasets) | CSV | Detection results |
| `detection_performance_all_datasets.xlsx` | YOLO comparison (formatted) | Excel | Paper tables |
| `iou_thresholds_analysis.csv` | mAP@50, mAP@75, mAP@50:95 | CSV | IoU analysis |
| `detection_summary.xlsx` | Detection summary (Excel) | Excel | Quick reference |

**Example: detection_performance_all_datasets.csv**
```csv
Dataset,Model,mAP@50,mAP@50-95,Precision,Recall,F1_Score
iml_lifecycle,YOLO10,0.9385,0.7892,0.9045,0.9428,0.9232
iml_lifecycle,YOLO11,0.9457,0.8006,0.9160,0.9510,0.9331
iml_lifecycle,YOLO12,0.9412,0.7956,0.9102,0.9469,0.9281
...
```

**Regeneration**:
```bash
# Full regeneration (includes all detection tables)
python scripts/publication/generate_all_publication_outputs.py
```

**Source**: Extracted from `results/optA_*/experiments/experiment_*/det_*/results.csv`

---

### 6. tables/statistics/ (4 files)

**Purpose**: Dataset statistics and augmentation analysis

| File | Description | Format | Use In |
|------|-------------|--------|--------|
| `dataset_statistics_all.csv` | Augmentation effects (all datasets) | CSV | Dataset section |
| `class_distribution.csv` | Class balance analysis | CSV | Data description |
| `training_validation_split.csv` | Train/val/test split info | CSV | Experimental setup |
| `augmentation_multipliers.csv` | Augmentation multiplier effects | CSV | Augmentation analysis |

**Example: dataset_statistics_all.csv**
```csv
Dataset,Original_Train,Detection_Aug,Classification_Aug,Det_Multiplier,Cls_Multiplier
iml_lifecycle,218,956,765,4.4x,3.5x
mp_idb_species,146,640,512,4.4x,3.5x
mp_idb_stages,146,640,512,4.4x,3.5x
```

**Regeneration**:
```bash
# Full regeneration (includes all statistics tables)
python scripts/publication/generate_all_publication_outputs.py
```

**Source**: Extracted from `results/optA_*/experiments/experiment_*/analysis_dataset_statistics/`

---

## 📋 Metadata Tracking (_metadata.json)

### Purpose

Tracks generation metadata and ensures data integrity:
- When files were generated
- Which experiment results were used
- File checksums for integrity verification
- Missing or corrupted file detection

### Structure

```json
{
  "generated_at": "2025-10-12T10:30:00.123456",
  "pipeline_version": "v2.0.0",
  "experiment_id": "optA_20251012_103000",
  "experiment_path": "results/optA_20251012_103000",
  "
_info": {
    "total_count": 44,
    "figures": {
      "count": 31,
      "categories": {
        "pipeline_diagrams": 5,
        "augmentation": 21,
        "performance": 8
      }
    },
    "tables": {
      "count": 12,
      "categories": {
        "classification": 4,
        "detection": 4,
        "statistics": 4
      }
    }
  },
  "files": {
    "figures/pipeline_diagrams/pipeline_architecture_publication.png": {
      "size_bytes": 245678,
      "checksum_md5": "a3b2c1d4e5f6...",
      "generated_at": "2025-10-12T10:30:15"
    },
    ...
  },
  "verification": {
    "all_files_present": true,
    "integrity_check": "passed",
    "last_verified": "2025-10-12T10:35:00",
    "missing_files": [],
    "corrupted_files": []
  }
}
```

### Usage

```bash
# View metadata
cat luaran/auto_generated/_metadata.json | jq '.'

# Check generation time
cat luaran/auto_generated/_metadata.json | jq '.generated_at'

# Verify integrity
python scripts/publication/verify_publication_data.py
```

---

## ✅ Data Integrity Verification

### Automatic Verification

**Included in regeneration**:
```bash
python scripts/publication/generate_all_publication_outputs.py

# Output includes verification:
# ✅ Generated 31 figures
# ✅ Exported 12 tables
# ✅ Created metadata.json
# ✅ Verification: All files present and valid
```

### Manual Verification

```bash
# Full verification (checks existence, integrity, data consistency)
python scripts/publication/verify_publication_data.py

# Check for missing files only
python scripts/publication/verify_publication_data.py --check-missing

# Verify data consistency only
python scripts/publication/verify_publication_data.py --verify-data

# Verbose output
python scripts/publication/verify_publication_data.py --verbose
```

**Verification checks**:
1. ✅ All expected files present
2. ✅ No corrupted files (checksum validation)
3. ✅ Data consistency (values match source experiments)
4. ✅ File sizes within expected ranges
5. ✅ Timestamps are recent and valid

---

## 🔄 When to Regenerate

### Required Regeneration

**After these events, regeneration is REQUIRED**:
1. ✅ New pipeline experiments completed
2. ✅ Experiment parameters changed (epochs, batch size, learning rate)
3. ✅ New models or datasets added
4. ✅ Before paper/report submission (to ensure latest data)
5. ✅ After bug fixes in pipeline code

### Optional Regeneration

**Consider regenerating in these cases**:
1. 🔄 Figure formatting improvements (higher resolution, better colors)
2. 🔄 Table structure changes (new columns, different ordering)
3. 🔄 Metadata updates (additional tracking fields)

### Verification Without Regeneration

**When outputs are already up-to-date**:
```bash
# Just verify, don't regenerate
python scripts/publication/verify_publication_data.py
```

---

## 🚨 Common Issues & Solutions

### Issue 1: Regeneration fails with "Experiment not found"

**Cause**: No experiment results available

**Solution**:
```bash
# Check if experiments exist
ls -la results/optA_*/

# If empty, run pipeline first
python main_pipeline.py --dataset all

# Then regenerate
python scripts/publication/generate_all_publication_outputs.py
```

### Issue 2: Missing figures or tables

**Cause**: Partial regeneration or script errors

**Solution**:
```bash
# Force full regeneration
python scripts/publication/generate_all_publication_outputs.py --force

# Verify what's missing
python scripts/publication/verify_publication_data.py --check-missing
```

### Issue 3: Metadata shows "corrupted files"

**Cause**: File was manually edited or transfer error

**Solution**:
```bash
# Regenerate corrupted files
python scripts/publication/generate_all_publication_outputs.py --fix-corrupted

# Or regenerate everything
python scripts/publication/generate_all_publication_outputs.py
```

### Issue 4: Figures not updating in paper

**Cause**: Cached figures in document viewer

**Solution**:
1. Close all viewers (PDF readers, Word, etc.)
2. Regenerate figures
3. Reopen document
4. If using Pandoc, add `--resource-path` flag

### Issue 5: Table data doesn't match experiments

**Cause**: Wrong experiment used for regeneration

**Solution**:
```bash
# Specify correct experiment
python scripts/publication/generate_all_publication_outputs.py \
  --experiment-id optA_20251012_103000

# Verify data consistency
python scripts/publication/verify_publication_data.py --verify-data
```

---

## 📊 Performance & Storage

### Generation Performance

| Category | Files | Time | Bottleneck |
|----------|-------|------|------------|
| Pipeline Diagrams | 5 | ~30s | Diagram rendering |
| Augmentation Figures | 21 | ~2m | Image processing |
| Performance Figures | 8 | ~1m | Plot generation |
| Classification Tables | 4 | ~30s | CSV export |
| Detection Tables | 4 | ~30s | Excel formatting |
| Statistics Tables | 4 | ~30s | Data aggregation |
| **TOTAL** | **44** | **~5m** | Image processing |

**Optimization tips**:
- Use `--figures-only` or `--tables-only` for faster partial updates
- Run on GPU-enabled machine for faster GradCAM generation
- Use `--parallel` flag for multi-threaded generation (future feature)

### Storage Requirements

| Category | Files | Size (Approx) |
|----------|-------|---------------|
| Pipeline Diagrams | 5 | 2-3 MB |
| Augmentation Figures | 21 | 8-12 MB |
| Performance Figures | 8 | 4-6 MB |
| Tables (CSV) | 8 | 100-200 KB |
| Tables (Excel) | 4 | 200-300 KB |
| Metadata | 1 | 20-50 KB |
| **TOTAL** | **44** | **~15-20 MB** |

**Storage optimization**:
- Figures use optimized PNG compression
- Tables use efficient CSV format
- Excel files are minimal (only formatted tables)
- Metadata is compact JSON

---

## 🔗 Integration with Papers

### Referencing Auto-Generated Files in Papers

**In Markdown papers** (`hand_created/papers/*.md`):
```markdown
<!-- Figures -->
![Pipeline Architecture](../../auto_generated/figures/pipeline_diagrams/pipeline_architecture_publication.png)

![Augmentation - IML Lifecycle](../../auto_generated/figures/augmentation/augmentation_iml_lifecycle_combined.png)

<!-- Tables -->
See classification results in Table 9: `../../auto_generated/tables/classification/table9_focal_loss.csv`

**Table 1**: Detection performance (source: `../../auto_generated/tables/detection/detection_performance_all_datasets.csv`)
```

**In DOCX/PDF exports**:
```bash
# Pandoc with resource paths
pandoc hand_created/papers/JICEST_Paper.md \
  --resource-path=.:../../auto_generated/figures:../../auto_generated/tables \
  -o hand_created/papers/exports/JICEST_Paper.docx
```

### Embedding Tables in Papers

**Option 1: Markdown tables** (copy from CSV):
```markdown
| Dataset | Model | mAP@50 | mAP@50-95 |
|---------|-------|--------|-----------|
| IML Lifecycle | YOLO11 | 0.9457 | 0.8006 |
| MP-IDB Species | YOLO11 | 0.9288 | 0.5575 |
```

**Option 2: Direct CSV include** (Pandoc filter):
```markdown
<!-- Include table from CSV -->
```{.include}
../../auto_generated/tables/detection/detection_performance_all_datasets.csv
```
```

**Option 3: Excel tables** (import in Word):
1. Open `hand_created/papers/exports/JICEST_Paper.docx`
2. Insert → Object → From File
3. Select `../../auto_generated/tables/detection/detection_summary.xlsx`

---

## 📚 Additional Resources

- **Main Project Documentation**: See `../../CLAUDE.md` for full pipeline guide
- **Luaran Overview**: See `../README.md` for directory structure
- **Manual Content Guide**: See `../hand_created/README.md` for paper writing best practices
- **Regeneration Scripts**: See `../../scripts/publication/` for all generation scripts

---

## 🔄 Change Log

### 2025-10-12: Initial Auto-Generation Structure
- ✅ Created auto_generated/ directory
- ✅ Implemented one-command regeneration
- ✅ Added metadata tracking and integrity verification
- ✅ Generated 44 files (31 figures + 12 tables)

---

*Last Updated: 2025-10-12*
*Maintained by: Automated Pipeline Scripts*
*⚠️ DO NOT EDIT FILES IN THIS DIRECTORY MANUALLY*
