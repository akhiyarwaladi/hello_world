# 🧹 CODEBASE CLEANUP ANALYSIS - FILE REDUNDANCY REPORT

**Date**: 2025-10-11
**Purpose**: Identify redundant files and organize codebase structure

---

## 📊 CURRENT CODEBASE STRUCTURE

### Root Directory Files (23 Python + MD files)

#### **Active/Core Files** ✅ (KEEP)
1. `run_multiple_models_pipeline_OPTION_A.py` - **MAIN PIPELINE** (only actively used)
2. `CLAUDE.md` - Project documentation
3. `run_baseline_comparison.py` - Baseline experiments

#### **Redundant Pipeline Diagram Scripts** ⚠️ (5 FILES - KEEP ONLY 1)
1. `create_pipeline_diagram.py` - Original version
2. `create_pipeline_diagram_v2.py` - Version 2
3. `create_pipeline_final.py` - "Final" version
4. `create_pipeline_clean.py` - "Clean" version
5. `create_pipeline_diagram_publication.py` - **CURRENT VERSION** ✅ (Keep this one)

**Status**:
- ✅ **KEEP**: `create_pipeline_diagram_publication.py` (latest, generates publication-quality diagrams)
- ❌ **DELETE**: Other 4 versions (obsolete)

---

#### **One-Time Fix Scripts** 🔧 (ALREADY EXECUTED - CAN DELETE)
1. `fix_all_code_switching.py` - Fixed Bahasa/English mixing
2. `fix_iml_removal.py` - Removed IML dataset references
3. `fix_language_consistency.py` - Language consistency fixes
4. `update_to_mp_idb_only.py` - Updated to MP-IDB only

**Status**:
- ⚠️ **ARCHIVE or DELETE** - These were one-time operations, already executed
- If you want history, move to `archive/` folder

---

#### **Figure/Document Generation Scripts** 📊 (ONE-TIME USE)
1. `add_figure_narratives.py` - Added figure descriptions (done)
2. `check_figure_mentions.py` - Verified figure mentions (done)
3. `generate_comprehensive_consolidated_analysis.py` - Generated consolidated analysis (done)
4. `generate_docx_from_markdown.py` - Convert MD to DOCX
5. `restructure_laporan_kemajuan.py` - Restructured progress report (done)

**Status**:
- ✅ **KEEP**: `generate_docx_from_markdown.py` (useful for future conversions)
- ⚠️ **ARCHIVE**: Others (one-time use, already executed)

---

#### **Documentation Files** 📝 (KEEP)
1. `CLAUDE.md` - ✅ Main project instructions
2. `FINAL_VERIFICATION.md` - ✅ Verification checklist
3. `HOWTO_ADD_NEW_LOSS_OR_MODEL.md` - ✅ Developer guide
4. `HOWTO_BATCH_GENERATE_ALL_FIGURES.md` - ✅ Figure generation guide
5. `HOWTO_GENERATE_AUGMENTATION_FIGURES.md` - ✅ Augmentation guide
6. `HOWTO_GENERATE_FIGURES_DETECTION_CLASSIFICATION.md` - ✅ Figure guide

**Status**: All useful documentation - **KEEP ALL**

---

## 📁 LUARAN/ DIRECTORY ANALYSIS

### Active Documents ✅ (KEEP)
1. `JICEST_Paper.md` - **PRIMARY PAPER** (conference submission)
2. `Draft_Journal_Q1_IEEE_TMI.md` - **JOURNAL DRAFT** (Q1 submission)
3. `Laporan_Kemajuan.md` - Progress report (Indonesian)

### Redundant Reports ⚠️ (MULTIPLE VERSIONS)
1. `Laporan_Kemajuan.md` - Main version ✅
2. `Laporan_Kemajuan_RINGKAS.md` - Condensed version
3. `Laporan_Kemajuan_BACKUP.md` - Backup (likely exists)
4. `Laporan_Kemajuan_BEFORE_RINGKAS.md` - Before condensing (likely exists)

**Status**:
- ✅ **KEEP**: `Laporan_Kemajuan.md` (main version)
- ⚠️ **ARCHIVE**: Other versions (backups, not needed anymore)

### luaran/figures/ ✅ (WELL ORGANIZED)
- All figures properly generated
- `enhance_pipeline_figure.py` - Enhancement script ✅ (Keep)
- Multiple versions of pipeline architecture (already cleaned today)

### luaran/tables/ ✅ (WELL ORGANIZED)
- All tables properly extracted
- Verification reports complete
- No redundancy detected

---

## 📊 REDUNDANCY SUMMARY

### Critical Redundancy (High Priority Cleanup)

| Category | Total Files | Keep | Archive/Delete | Redundancy % |
|----------|------------|------|----------------|-------------|
| Pipeline Diagram Scripts | 5 | 1 | 4 | **80%** |
| Fix Scripts (one-time) | 4 | 0 | 4 | **100%** |
| Figure Generation (one-time) | 4 | 1 | 3 | **75%** |
| Laporan Kemajuan versions | 4 | 1 | 3 | **75%** |
| **TOTAL** | **17** | **3** | **14** | **82%** |

### Files to Keep (Core Active Files)
1. `run_multiple_models_pipeline_OPTION_A.py` - Main pipeline
2. `create_pipeline_diagram_publication.py` - Latest diagram generator
3. `generate_docx_from_markdown.py` - Useful converter
4. `run_baseline_comparison.py` - Baseline experiments
5. All `CLAUDE.md`, `HOWTO_*.md`, `FINAL_VERIFICATION.md` - Documentation
6. All files in `scripts/` directory - Active training/analysis
7. All files in `luaran/` except redundant versions

---

## 🚨 RECOMMENDED CLEANUP ACTIONS

### Phase 1: Archive Old Versions (Safe - Can Restore Later)

Create archive folder:
```bash
mkdir archive
mkdir archive/pipeline_diagrams
mkdir archive/one_time_fixes
mkdir archive/figure_generators
mkdir archive/laporan_backup
```

Move files to archive:
```bash
# Pipeline diagrams (keep only publication version)
mv create_pipeline_diagram.py archive/pipeline_diagrams/
mv create_pipeline_diagram_v2.py archive/pipeline_diagrams/
mv create_pipeline_final.py archive/pipeline_diagrams/
mv create_pipeline_clean.py archive/pipeline_diagrams/

# One-time fix scripts
mv fix_*.py archive/one_time_fixes/
mv update_to_mp_idb_only.py archive/one_time_fixes/

# One-time figure generators
mv add_figure_narratives.py archive/figure_generators/
mv check_figure_mentions.py archive/figure_generators/
mv generate_comprehensive_consolidated_analysis.py archive/figure_generators/
mv restructure_laporan_kemajuan.py archive/figure_generators/

# Laporan backup versions (if exist)
mv luaran/Laporan_Kemajuan_RINGKAS.md archive/laporan_backup/
mv luaran/Laporan_Kemajuan_BACKUP.md archive/laporan_backup/ 2>/dev/null
mv luaran/Laporan_Kemajuan_BEFORE_RINGKAS.md archive/laporan_backup/ 2>/dev/null
```

**Space Savings**: ~14 redundant files → cleaner root directory

---

### Phase 2: Delete Truly Redundant Files (If Confident)

After confirming everything works, you can delete archived files:
```bash
# Only if you're 100% sure you won't need them
rm -rf archive/
```

**⚠️ WARNING**: Only do this after thorough testing!

---

## 📁 RECOMMENDED FINAL STRUCTURE

```
hello_world/
├── CLAUDE.md                                    # ✅ Main documentation
├── FINAL_VERIFICATION.md                        # ✅ Verification checklist
├── HOWTO_*.md                                   # ✅ Developer guides (6 files)
│
├── run_multiple_models_pipeline_OPTION_A.py     # ✅ MAIN PIPELINE
├── run_baseline_comparison.py                   # ✅ Baseline experiments
├── create_pipeline_diagram_publication.py       # ✅ Latest diagram generator
├── generate_docx_from_markdown.py               # ✅ Useful converter
│
├── scripts/                                     # ✅ All active scripts
│   ├── training/                                # Training scripts
│   ├── analysis/                                # Analysis scripts
│   ├── data_setup/                              # Dataset setup
│   └── visualization/                           # Visualization scripts
│
├── luaran/                                      # ✅ Research outputs
│   ├── JICEST_Paper.md                          # ✅ Conference paper
│   ├── Draft_Journal_Q1_IEEE_TMI.md             # ✅ Journal draft
│   ├── Laporan_Kemajuan.md                      # ✅ Progress report
│   ├── figures/                                 # All figures
│   ├── tables/                                  # All tables
│   └── FIGURE_SELECTION_RANKING_TOP_IMAGES.md   # Image ranking
│
├── data/                                        # Dataset files
├── results/                                     # Experiment results
├── utils/                                       # Utility modules
│
└── archive/                                     # ⚠️ Archived old files
    ├── pipeline_diagrams/                       # Old diagram scripts
    ├── one_time_fixes/                          # Fix scripts (executed)
    ├── figure_generators/                       # One-time generators
    └── laporan_backup/                          # Report backups
```

---

## 🎯 BENEFITS OF CLEANUP

### Before Cleanup:
- **Root directory**: 23+ Python/MD files (confusing!)
- **Redundancy**: 82% of generation scripts are duplicates
- **Hard to find**: Main pipeline buried among old scripts

### After Cleanup:
- **Root directory**: ~10 essential files (clear!)
- **No redundancy**: Only active/useful scripts remain
- **Easy to find**: Main pipeline immediately visible
- **Clean structure**: Logical organization

---

## ✅ IMMEDIATE ACTIONS (Minimal Risk)

1. **Create Archive Folder** (5 minutes)
   ```bash
   mkdir archive
   mkdir archive/{pipeline_diagrams,one_time_fixes,figure_generators,laporan_backup}
   ```

2. **Move Redundant Pipeline Diagrams** (Safe to archive)
   ```bash
   mv create_pipeline_diagram.py archive/pipeline_diagrams/
   mv create_pipeline_diagram_v2.py archive/pipeline_diagrams/
   mv create_pipeline_final.py archive/pipeline_diagrams/
   mv create_pipeline_clean.py archive/pipeline_diagrams/
   ```

3. **Move One-Time Fix Scripts** (Already executed)
   ```bash
   mv fix_all_code_switching.py archive/one_time_fixes/
   mv fix_iml_removal.py archive/one_time_fixes/
   mv fix_language_consistency.py archive/one_time_fixes/
   mv update_to_mp_idb_only.py archive/one_time_fixes/
   ```

4. **Update CLAUDE.md** (Document changes)
   - Add note about archived files
   - Update file structure documentation

---

## 🚫 DO NOT TOUCH (Critical Active Files)

1. **NEVER DELETE**:
   - `run_multiple_models_pipeline_OPTION_A.py` - Main pipeline
   - `CLAUDE.md` - Project documentation
   - Anything in `scripts/` directory
   - Anything in `data/`, `results/`, `utils/`

2. **NEVER ARCHIVE**:
   - Current papers in `luaran/`
   - Tables and figures in `luaran/tables/`, `luaran/figures/`
   - Any `HOWTO_*.md` documentation

---

## 📝 CLEANUP CHECKLIST

- [ ] Create `archive/` folder structure
- [ ] Move 4 old pipeline diagram scripts to archive
- [ ] Move 4 one-time fix scripts to archive
- [ ] Move 3 one-time figure generators to archive
- [ ] Move backup laporan versions to archive
- [ ] Update CLAUDE.md with new structure
- [ ] Test main pipeline still works
- [ ] Verify all documentation accessible
- [ ] (Optional) Delete archive/ after 1 month if not needed

---

## 💡 MAINTENANCE TIPS

1. **Before Creating New Files**:
   - Check if similar file already exists
   - Use version control (git) instead of creating v2, v3, etc.

2. **Naming Convention**:
   - Prefix with purpose: `generate_`, `fix_`, `analyze_`
   - Include date for one-time scripts: `fix_data_20250101.py`

3. **Documentation**:
   - Add file purpose at top of each script
   - Mark as "[ONE-TIME]" if not reusable

4. **Regular Cleanup**:
   - Archive one-time scripts after execution
   - Review root directory monthly
   - Keep only active files visible

---

**Status**: ✅ Analysis Complete - Ready for Cleanup
**Risk Level**: 🟢 LOW (all changes are reversible via archive/)
**Estimated Time**: 15-20 minutes for complete cleanup
