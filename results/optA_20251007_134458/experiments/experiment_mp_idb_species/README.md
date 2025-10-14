# Option A Pipeline Results: optA_20251007_134458_mp_idb_species

## Summary
- **Generated**: 2025-10-07 17:43:52
- **Pipeline Type**: Option A - Shared Classification Architecture
- **Total Components**: 0

## Folder Structure (Legacy)
- `detection/` - Detection model results and weights (independent)
- `classification/` - Classification model results and weights (SHARED)
- `crop_data/` - Generated crop datasets (SHARED, single instance)
- `analysis/` - Analysis results (separate detection vs classification)
- `master_summary.json` - Detailed summary

## Key Efficiency Improvements
- **~70% Storage Reduction**: Classification models trained once, not per detection model
- **~60% Training Time Reduction**: Ground truth crops generated once
- **No Duplication**: Clean separation between detection and classification stages
- **Shared Architecture**: All detection models use same ground truth crops and classification models

## Architecture Benefits
This archive uses Option A architecture where:
1. Detection models are trained independently
2. Ground truth crops are generated ONCE and shared
3. Classification models are trained ONCE and shared
4. Analysis is done separately for detection vs classification

This eliminates the storage and training time waste of the original architecture.
