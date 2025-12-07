# Data Setup Scripts

## Overview

This directory contains scripts for preparing datasets for the malaria detection pipeline.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SETUP PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GENERAL PIPELINE (Optional - for custom datasets):           │
│   01 → 02 → 03 → 04 → 05 → 06                                  │
│                                                                 │
│   DATASET-SPECIFIC SETUP (Choose ONE per dataset):             │
│   ├── 07_setup_kaggle_species   → MP-IDB Species dataset       │
│   ├── 08_setup_lifecycle        → IML Lifecycle dataset        │
│   ├── 09_setup_kaggle_stage     → MP-IDB Stages dataset        │
│   └── 10_setup_md2019_stage     → MD-2019 Stages dataset       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Scripts Description

### General Pipeline (01-06)

These scripts are for custom dataset processing. **For standard datasets, use scripts 07-10 instead.**

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_download_datasets.py` | Download raw datasets | URLs | `data/raw/` |
| `02_preprocess_data.py` | Basic preprocessing | Raw images | Cleaned images |
| `03_integrate_datasets.py` | Merge multiple sources | Multiple datasets | Single dataset |
| `04_convert_to_yolo.py` | Convert to YOLO format | Various formats | YOLO annotations |
| `05_augment_data.py` | Apply augmentations | Images + labels | Augmented data |
| `06_split_dataset.py` | Train/val/test split | Full dataset | Split folders |

### Dataset-Specific Scripts (07-10)

**These are the primary scripts for setting up standard datasets:**

| Script | Dataset | Source | Classes |
|--------|---------|--------|---------|
| `07_setup_kaggle_species_for_pipeline.py` | MP-IDB Species | Kaggle | 4 species |
| `08_setup_lifecycle_for_pipeline.py` | IML Lifecycle | GitHub | 4 stages |
| `09_setup_kaggle_stage_for_pipeline.py` | MP-IDB Stages | Kaggle | 4 stages |
| `10_setup_md2019_stage_for_pipeline.py` | MD-2019 Stages | Manual | 3 stages |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `generate_high_quality_crops.py` | Generate high-resolution (512px) crops |
| `upscale_existing_crops.py` | Upscale existing 224px crops to higher resolution |

---

## Usage

### Quick Setup (Recommended)

For a new PC, run the dataset-specific scripts directly:

```bash
# Setup all 4 datasets with 60/20/20 split
python scripts/data_setup/08_setup_lifecycle_for_pipeline.py \
    --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20

python scripts/data_setup/07_setup_kaggle_species_for_pipeline.py \
    --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20

python scripts/data_setup/09_setup_kaggle_stage_for_pipeline.py \
    --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20

python scripts/data_setup/10_setup_md2019_stage_for_pipeline.py \
    --train-ratio 0.60 --val-ratio 0.20 --test-ratio 0.20
```

### Expected Output

After running, you should have:

```
data/
├── raw/                          # Downloaded raw data
│   ├── malaria_lifecycle/        # IML dataset
│   ├── kaggle_dataset/           # MP-IDB dataset
│   └── md_2019/                  # MD-2019 dataset
│
└── processed/                    # Detection-ready data
    ├── lifecycle/                # 313 images (60/20/20 split)
    │   ├── train/images/         # 186 images
    │   ├── val/images/           # 64 images
    │   ├── test/images/          # 63 images
    │   └── data.yaml
    ├── species/                  # 209 images
    ├── stages/                   # 209 images
    └── md_2019_stages/           # 813 images
```

---

## Common Arguments

All dataset-specific scripts (07-10) support:

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-ratio` | 0.60 | Training set ratio |
| `--val-ratio` | 0.20 | Validation set ratio |
| `--test-ratio` | 0.20 | Test set ratio |
| `--output` | auto | Output directory |
| `--seed` | 42 | Random seed for reproducibility |

---

## Dependencies

- **Scripts 07-09**: Require `kaggle` CLI for dataset download
- **Script 10**: Requires manual download of MD-2019 dataset

### Kaggle Setup

```bash
# Install kaggle CLI
pip install kaggle

# Place kaggle.json in ~/.kaggle/ or project root
# Download from: https://www.kaggle.com/settings → API → Create New Token
```

---

## Troubleshooting

### "Dataset not found"
- Check internet connection
- Verify Kaggle credentials (`kaggle.json`)
- For MD-2019: manually download and extract to `data/raw/md_2019/`

### "Path errors"
```bash
# Fix data.yaml paths
python fix_data_yaml_paths.py
```

### "Wrong split ratio"
- Re-run the setup script with correct `--train-ratio`, `--val-ratio`, `--test-ratio`

---

*Last Updated: 2025-12-07*
