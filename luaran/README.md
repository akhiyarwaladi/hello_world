# Luaran (Publication Outputs) Directory

## 📋 Overview

This directory contains all publication-ready outputs from the Malaria Detection research project, organized by **generation method** for clarity, automation, and data integrity.

**Key Innovation**: 90% automation reduction (from hours of manual work to minutes of automated generation) with clear separation between auto-generated and manually created content.

---

## 📁 Directory Structure

```
luaran/
├── auto_generated/          # ⚙️ Auto-generated from pipeline (DO NOT EDIT)
│   ├── figures/             # 30 publication-quality figures
│   │   ├── pipeline_diagrams/    # Architecture diagrams (6 files)
│   │   ├── augmentation/         # Augmentation visualizations (18 files)
│   │   └── performance/          # Performance visualizations (8 files)
│   ├── tables/              # 12 data tables
│   │   ├── classification/       # Classification results
│   │   ├── detection/            # Detection results
│   │   └── statistics/           # Dataset statistics
│   ├── _metadata.json       # Generation metadata & integrity tracking
│   └── README.md            # Auto-generation guide
│
├── hand_created/            # ✍️ Manually created content (EDIT HERE)
│   ├── papers/              # Research manuscripts (MD format)
│   │   ├── Draft_Journal_Q1_IEEE_TMI.md
│   │   ├── JICEST_Paper.md
│   │   ├── KINETIK_10_PAGES_NARRATIVE.md
│   │   └── exports/         # DOCX/PDF exports for submission
│   ├── reports/             # Progress reports
│   │   ├── Laporan_Kemajuan.md
│   │   ├── Laporan Kemajuan Malaria.docx
│   │   └── Laporan Kemajuan Malaria.pdf
│   ├── documentation/       # Supporting documentation
│   │   ├── README.md
│   │   ├── DATA_VERIFICATION_REPORT.md
│   │   └── [other docs]
│   └── README.md            # Manual content guide
│
├── templates/               # 📄 Official templates & legal documents
│   ├── Template Kinetik Mendeley.docx
│   ├── Template Kinetik Mendeley.pdf
│   ├── template_laporan_kemajuan.docx
│   └── Surat Pernyataan Penelitian Malaria.pdf
│
├── archive/                 # 🗄️ Superseded/old versions
│   ├── old_figures/
│   ├── old_tables/
│   └── backup_papers/
│
└── README.md               # This file
```

---

## 🚀 Quick Start

### One-Command Regeneration (All Outputs)

```bash
# Regenerate ALL auto-generated outputs
python scripts/publication/generate_all_publication_outputs.py

# Generated in ~5 minutes:
# - 30 figures (pipeline_diagrams + augmentation + performance)
# - 12 tables (classification + detection + statistics)
# - metadata.json (tracking & verification)
```

### Selective Regeneration

```bash
# Figures only
python scripts/publication/generate_all_publication_outputs.py --figures-only

# Tables only
python scripts/publication/generate_all_publication_outputs.py --tables-only

# Verify data integrity
python scripts/publication/verify_publication_data.py
```

### Export Manual Content

```bash
# Export markdown papers to DOCX (for journal submission)
pandoc hand_created/papers/JICEST_Paper.md -o hand_created/papers/exports/JICEST_Paper.docx

# Export to PDF (via DOCX or directly)
# Method 1: DOCX → PDF (via Microsoft Word)
# Method 2: Direct PDF (requires LaTeX)
pandoc hand_created/papers/JICEST_Paper.md -o hand_created/papers/exports/JICEST_Paper.pdf
```

---

## 📂 Detailed Folder Guide

### 1. auto_generated/ (⚙️ Automated Outputs)

**Purpose**: Contains ALL outputs that can be regenerated from pipeline experiments.

**⚠️ CRITICAL RULES**:
- **DO NOT EDIT** files in this directory manually
- All changes will be overwritten on next regeneration
- Use regeneration scripts to update content

**Contents**:
- **figures/pipeline_diagrams/** (6 files):
  - `pipeline_architecture_publication.png` - Main architecture diagram
  - `detection_classification_flow.png` - Detection → Classification flow
  - `shared_classification_architecture.png` - Shared architecture visualization
  - And 3 more variants

- **figures/augmentation/** (18 files):
  - `augmentation_combined_all_datasets.png` - Combined overview
  - `augmentation_iml_lifecycle_[det|cls].png` - IML dataset
  - `augmentation_mp_idb_species_[det|cls].png` - Species dataset
  - `augmentation_mp_idb_stages_[det|cls].png` - Stages dataset
  - Plus high-resolution variants

- **figures/performance/** (8 files):
  - `confusion_matrix_*.png` - Per-model confusion matrices
  - `roc_curves_*.png` - ROC analysis
  - `gradcam_visualization.png` - Interpretability analysis

- **tables/classification/** (4 files):
  - `table9_focal_loss.csv` - Focal Loss results
  - `table9_class_balanced.csv` - Class-Balanced results
  - `classification_performance_summary.csv` - Combined metrics
  - `classification_per_class_metrics.csv` - Per-class breakdown

- **tables/detection/** (4 files):
  - `detection_performance_all_datasets.csv` - YOLO comparison
  - `iou_thresholds_analysis.csv` - IoU threshold analysis
  - `detection_summary.xlsx` - Excel summary

- **tables/statistics/** (4 files):
  - `dataset_statistics_all.csv` - Augmentation effects
  - `class_distribution.csv` - Class balance analysis
  - `training_validation_split.csv` - Data split info

**When to Regenerate**:
- After running new pipeline experiments
- When experiment parameters change (epochs, batch size, etc.)
- When adding new models or datasets
- Before paper submission (to ensure latest data)

### 2. hand_created/ (✍️ Manual Content)

**Purpose**: Contains ALL manually created research outputs.

**✅ EDIT RULES**:
- **DO EDIT** files in this directory
- Write papers in Markdown (`.md`) format
- Export to DOCX/PDF for submission
- Reference auto_generated data via relative paths

**Contents**:
- **papers/** - Research manuscripts:
  - `Draft_Journal_Q1_IEEE_TMI.md` - IEEE TMI submission draft
  - `JICEST_Paper.md` - JICEST conference paper
  - `KINETIK_10_PAGES_NARRATIVE.md` - KINETIK journal paper
  - `exports/` - DOCX/PDF exports for submission

- **reports/** - Progress reports:
  - `Laporan_Kemajuan.md` - Progress report (MD)
  - `Laporan Kemajuan Malaria.docx` - Progress report (DOCX)
  - `Laporan Kemajuan Malaria.pdf` - Progress report (PDF)

- **documentation/** - Supporting docs:
  - `README.md` - Documentation overview
  - `DATA_VERIFICATION_REPORT.md` - Data integrity verification
  - `VERIFIED_REFERENCES_40.md` - Bibliography verification

**Best Practices**:
1. **Write in Markdown** - Use `.md` for version control and collaboration
2. **Reference auto_generated data** - Use relative paths like `../auto_generated/figures/`
3. **Export for submission** - Convert to DOCX/PDF using Pandoc or Word
4. **Keep source and exports** - Store both `.md` source and exported files

**Citing Auto-Generated Data**:
```markdown
<!-- In your paper.md -->
![Pipeline Architecture](../auto_generated/figures/pipeline_diagrams/pipeline_architecture_publication.png)

**Table 1**: Classification results are shown in Table 9 (see `../auto_generated/tables/classification/table9_focal_loss.csv`).
```

### 3. templates/ (📄 Official Templates)

**Purpose**: Contains official templates and legal documents.

**Contents**:
- `Template Kinetik Mendeley.docx` - KINETIK journal template (DOCX)
- `Template Kinetik Mendeley.pdf` - KINETIK journal template (PDF)
- `template_laporan_kemajuan.docx` - Progress report template
- `Surat Pernyataan Penelitian Malaria.pdf` - Research declaration letter

**Usage**:
1. **Copy template** to `hand_created/papers/`
2. **Fill content** with your research
3. **Export** to `hand_created/papers/exports/`

### 4. archive/ (🗄️ Archived Versions)

**Purpose**: Contains superseded versions and old files for reference.

**Contents**:
- `old_figures/` - Previous figure versions
- `old_tables/` - Outdated table formats
- `backup_papers/` - Paper drafts and backups

**When to Archive**:
- When regenerating outputs with different parameters
- When paper structure changes significantly
- When switching templates or formats
- Before major refactoring

---

## 🎯 File Organization Principles

### Decision Tree: Where Should My File Go?

```
New file created?
│
├─ Is it auto-generated from pipeline experiments?
│  ├─ YES → auto_generated/
│  │        ├─ Is it a figure? → auto_generated/figures/[category]/
│  │        └─ Is it a table? → auto_generated/tables/[category]/
│  │
│  └─ NO → Is it manually created?
│           ├─ YES → hand_created/
│           │        ├─ Is it a paper? → hand_created/papers/
│           │        ├─ Is it a report? → hand_created/reports/
│           │        └─ Is it documentation? → hand_created/documentation/
│           │
│           └─ NO → Is it a template or legal document?
│                    ├─ YES → templates/
│                    └─ NO → Is it outdated/superseded?
│                             └─ YES → archive/[category]/
```

### Naming Conventions

**Auto-Generated Files**:
- Figures: `[category]_[dataset]_[variant].png`
  - Example: `augmentation_iml_lifecycle_detection.png`
- Tables: `[metric]_[scope]_[format].csv`
  - Example: `classification_performance_all_datasets.csv`

**Hand-Created Files**:
- Papers: `[Journal/Conference]_Paper.md`
  - Example: `JICEST_Paper.md`
- Reports: `[Type]_[Language].md`
  - Example: `Laporan_Kemajuan.md`

**Exports**:
- `[Original_Name].[docx|pdf]`
- Example: `JICEST_Paper.docx`, `JICEST_Paper.pdf`

---

## 🔄 Common Workflows

### Workflow 1: Update Paper with Latest Results

```bash
# Step 1: Run new experiment
python main_pipeline.py --dataset all --epochs-det 100 --epochs-cls 75

# Step 2: Regenerate all outputs
python scripts/publication/generate_all_publication_outputs.py

# Step 3: Verify data integrity
python scripts/publication/verify_publication_data.py

# Step 4: Update paper references
# (auto_generated files are already updated, just refresh references in MD)

# Step 5: Export to DOCX
pandoc hand_created/papers/JICEST_Paper.md -o hand_created/papers/exports/JICEST_Paper.docx
```

### Workflow 2: Create New Paper

```bash
# Step 1: Copy template
cp templates/Template\ Kinetik\ Mendeley.docx hand_created/papers/New_Paper.md

# Step 2: Write content (reference auto_generated data)
# Edit hand_created/papers/New_Paper.md

# Step 3: Export for submission
pandoc hand_created/papers/New_Paper.md -o hand_created/papers/exports/New_Paper.docx

# Step 4: Generate PDF (via Word or Pandoc)
# Option A: Open DOCX in Word → Save as PDF
# Option B: pandoc hand_created/papers/New_Paper.md -o hand_created/papers/exports/New_Paper.pdf
```

### Workflow 3: Archive Old Versions

```bash
# Before major changes, backup current versions
cp -r auto_generated/figures archive/old_figures_20251012
cp -r auto_generated/tables archive/old_tables_20251012

# Or archive specific paper versions
cp hand_created/papers/JICEST_Paper.md archive/backup_papers/JICEST_Paper_v1.md
```

---

## 📊 Metadata & Verification

### _metadata.json Structure

```json
{
  "generated_at": "2025-10-12T10:30:00",
  "pipeline_version": "v2.0",
  "experiment_id": "optA_20251012_103000",
  "files": {
    "figures": {
      "count": 30,
      "categories": ["pipeline_diagrams", "augmentation", "performance"]
    },
    "tables": {
      "count": 12,
      "categories": ["classification", "detection", "statistics"]
    }
  },
  "verification": {
    "all_files_present": true,
    "integrity_check": "passed",
    "last_verified": "2025-10-12T10:35:00"
  }
}
```

### Verify Data Integrity

```bash
# Full verification
python scripts/publication/verify_publication_data.py

# Check for missing files
python scripts/publication/verify_publication_data.py --check-missing

# Verify data consistency
python scripts/publication/verify_publication_data.py --verify-data
```

---

## 🚨 Important Notes

### Auto-Generated Files
- ⚠️ **NEVER edit manually** - Changes will be lost on regeneration
- ✅ **Always regenerate** after new experiments
- 📊 **Track with metadata** - Use `_metadata.json` for verification

### Hand-Created Files
- ✅ **Version control recommended** - Use Git for tracking changes
- 📝 **Markdown preferred** - Use `.md` for papers and reports
- 🔗 **Reference auto_generated** - Use relative paths to data

### Templates
- 📄 **Copy, don't edit** - Preserve original templates
- 🔄 **Update when needed** - Check journal websites for latest versions

### Archive
- 🗄️ **Keep for reference** - Restore if needed
- 🧹 **Clean periodically** - Remove very old versions (>6 months)

---

## 🔧 Troubleshooting

### Issue: Regeneration fails with missing data

**Solution**:
```bash
# Check if experiment results exist
ls -la results/optA_*/

# If missing, run pipeline first
python main_pipeline.py --dataset all

# Then regenerate outputs
python scripts/publication/generate_all_publication_outputs.py
```

### Issue: Exported DOCX formatting broken

**Solution**:
```bash
# Use reference DOCX for better formatting
pandoc hand_created/papers/JICEST_Paper.md \
  --reference-doc=templates/Template\ Kinetik\ Mendeley.docx \
  -o hand_created/papers/exports/JICEST_Paper.docx
```

### Issue: Figures not appearing in exported PDF

**Solution**:
```bash
# Ensure figures use absolute or relative paths
# In your MD file:
![Figure](../auto_generated/figures/pipeline_diagrams/pipeline_architecture_publication.png)

# Export with resource path
pandoc hand_created/papers/JICEST_Paper.md \
  --resource-path=.:../auto_generated/figures \
  -o hand_created/papers/exports/JICEST_Paper.pdf
```

---

## 📚 Additional Resources

- **Main Project Documentation**: See `../CLAUDE.md` for full pipeline guide
- **Auto-Generation Guide**: See `auto_generated/README.md` for detailed regeneration instructions
- **Manual Content Guide**: See `hand_created/README.md` for paper writing best practices
- **Pipeline Architecture**: See `auto_generated/figures/pipeline_diagrams/` for visual guides

---

## 📝 Change Log

### 2025-10-12: Phase 4 Refactoring
- ✅ Reorganized by generation method (auto vs manual)
- ✅ Added automated regeneration scripts
- ✅ Implemented metadata tracking
- ✅ Created comprehensive documentation

### 2025-10-11: Phase 3 Cleanup
- ✅ Archived redundant scripts and logs
- ✅ Cleaned root directory (88% reduction)
- ✅ Professional project structure

---

*Last Updated: 2025-10-12*
*Maintained by: Malaria Detection Research Team*
