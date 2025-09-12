# Data Directory - Context for Claude

## Purpose
Contains all datasets for malaria detection pipeline (directories gitignored for size).

## Structure
```
data/
├── raw/                    # Downloaded raw datasets (GITIGNORED)
│   ├── nih_cell/          # NIH Cell Images dataset  
│   ├── mp_idb/            # MP-IDB database
│   ├── bbbc041/           # BBBC041 dataset
│   ├── plasmoID/          # PlasmoID dataset
│   ├── iml/               # IML dataset
│   └── uganda/            # Uganda dataset
├── processed/             # Preprocessed images & metadata (GITIGNORED)
│   ├── images/            # ~27k+ processed images (currently growing)
│   └── processed_samples.csv  # Metadata (not created yet)
├── integrated/            # Unified dataset format
├── augmented/             # Augmented training data  
└── splits/                # Final train/val/test splits
```

## Current Status (Updated: December 12, 2024)
- ✅ **Raw data downloaded** - 6 datasets from various sources successfully downloaded
- ✅ **Initial preprocessing COMPLETED** - 56,754 images processed and integrated
- ❌ **Species mapping issue DISCOVERED & FIXED** - Only 2/6 classes had data initially
- 🔄 **Re-preprocessing ACTIVE** - ~15% complete with corrected species mapping
- 🔄 **Integration RE-RUNNING** - Processing updated data with proper species labels
- 📊 **Expected final dataset** - All 6 classes with balanced representation

## Data Sources
1. **NIH Cell Images** - 13,779 infected + 13,779 uninfected cells
2. **MP-IDB** - Whole slide images with species annotations
3. **BBBC041** - Additional microscopy images
4. **PlasmoID** - Species-specific parasite images
5. **IML** - Medical imaging dataset
6. **Uganda** - Local hospital dataset

## Dataset Classes (6 total)
- P_falciparum (Class 0)
- P_vivax (Class 1)  
- P_malariae (Class 2)
- P_ovale (Class 3)
- Mixed_infection (Class 4)
- Uninfected (Class 5)

## Important Notes
- All actual data files are **GITIGNORED** to prevent large commits
- Only .gitkeep files and metadata will be committed
- Pipeline automatically manages data flow between directories