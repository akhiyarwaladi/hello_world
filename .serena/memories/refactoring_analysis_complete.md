# Refactoring Analysis - Complete Mapping

## Pipeline Output Files (from Data Engineer Agent)

### Main Pipeline Outputs (results/ folder)
**Total per single dataset run**: ~140 files + ~1000 crop images (1-1.5 GB uncompressed)

**Stage Breakdown**:
1. **Detection** (45 files): 3 YOLO models × 15 files each
2. **Crops** (~1000 images + 1 CSV): Ground truth crops, shared
3. **Classification** (54 files): 6 models × 9 files each
4. **Analysis** (~45 files): 7 sub-stages
5. **Consolidated** (9 files): Multi-dataset only
6. **Archive** (4 summary + 1 ZIP)

**Critical Files for Publication**:
- table9_classification_pivot.xlsx (per-class metrics)
- detection_models_comparison.xlsx (YOLO comparison)
- dataset_statistics_summary.csv (augmentation effects)
- classification_performance_all_datasets.xlsx (cross-dataset)

## Luaran Folder Contents (from Data Scientist Agent)

### File Categories (70 total files)
- **Auto-Generated** (25 files, 36%):
  - 6 pipeline diagrams
  - 18 augmentation figures
  - 1 table (Table2)
  
- **Manual Extraction** (31 files, 44%):
  - 8 performance figures
  - 16 tables (CSV)
  - 6 document exports (DOCX/PDF)
  - 1 enhancement script
  
- **Hand-Created** (14 files, 20%):
  - 4 papers/reports (MD)
  - 6 documentation files
  - 4 templates/official docs

### Scripts Writing to luaran/
1. `scripts/analysis/generate_table2_from_experiment.py` → luaran/tables/
2. `scripts/visualization/generate_compact_augmentation_figures.py` → luaran/figures/
3. `scripts/visualization/generate_pipeline_architecture_diagram.py` → luaran/figures/
4. `luaran/figures/enhance_pipeline_figure.py` → luaran/figures/ (WRONG LOCATION)

### Critical Issues
1. **Data Integrity**: Draft_Journal_Q1_IEEE_TMI.md contains fabricated data
2. **Redundancy**: 4 superseded files (old MP-IDB only versions)
3. **Inconsistent Generation**: Only 1/17 tables auto-generated
4. **Wrong Location**: Enhancement script in luaran/figures/ instead of scripts/

## File Dependencies

### results/ → luaran/ Flow
**Current**: Manual extraction required
- User manually copies CSV from results/optA_*/consolidated_analysis/
- Reformats and saves to luaran/tables/
- **Problem**: Error-prone, time-consuming, inconsistent

**Should Be**: Automated export
- Pipeline auto-exports publication-ready files
- Consistent formatting, verified data
- One command regenerates everything

### Key Functions
- `create_centralized_zip()` (main_pipeline.py:190-339): Archives results/
- Results saved to: results/optA_[timestamp]/ (via ResultsManager)
- No automatic export to luaran/

## Refactoring Priorities

### HIGH (Critical Structure Issues)
1. Separate auto-generated vs manual outputs in luaran/
2. Create automated table generation from results/
3. Move enhancement script to correct location
4. Clean superseded files

### MEDIUM (Workflow Improvement)
5. Create unified publication export script
6. Automate performance figure generation
7. Add data verification step to exports

### LOW (Nice to Have)
8. Reorganize by paper sections
9. Version control for publication outputs
10. Template-based paper generation
