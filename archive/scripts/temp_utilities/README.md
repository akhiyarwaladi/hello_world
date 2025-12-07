# 📦 Archived Scripts

This folder contains **obsolete and deprecated** scripts that are no longer actively used in the project.

**Archived Date:** 2025-10-26

---

## 📋 Why These Scripts Are Archived

These scripts were used during development but have been replaced by improved versions or are one-time-use scripts that have already been executed.

---

## 📂 Archived Files

### Table Generation (Old Versions)
| Script | Reason for Archiving | Replaced By |
|--------|---------------------|-------------|
| `create_excel_tables.py` | Old combined version for all tables | Individual scripts in `luaran/templates/code/` |
| `create_classification_tables_multilevel.py` | Old format | `luaran/templates/code/generate_tables3456_classification.py` |
| `create_table_dataset_augmentation.py` | First version | `luaran/templates/code/generate_table1_dataset_augmentation.py` |
| `create_table1_dataset_augmentation.py` | Copied to templates/code | ✅ Now in `luaran/templates/code/` |
| `create_table1_horizontal.py` | Old horizontal format | Multi-level header format adopted |
| `create_table1_multilevel.py` | Early multi-level attempt | Final version improved |
| `create_classification_tables_final.py` | Copied to templates/code | ✅ Now in `luaran/templates/code/` |
| `create_table6_fair_comparison.py` | Early SOTA comparison | Final version with 5 papers |
| `create_table6_merged_format.py` | Merged format attempt | Final format chosen |
| `create_table6_opsi2_ringkas.py` | Option 2 (not chosen) | Final format selected |
| `create_table6_opsi3_merged.py` | Option 3 (not chosen) | Final format selected |
| `create_table6_sota_improved.py` | Improved version | Final version with 5 papers better |
| `create_table6_5papers_final.py` | Copied to templates/code | ✅ Now in `luaran/templates/code/` |

### Reference Renumbering (One-Time Use)
| Script | Reason for Archiving | Status |
|--------|---------------------|--------|
| `check_citation_order.py` | One-time verification | ✅ Task completed |
| `create_renumber_mapping.py` | Created reference mapping | ✅ Mapping created |
| `renumber_citations.py` | Renumbered all citations | ✅ All 141 citations updated |
| `reorder_references.py` | Reordered references section | ✅ References reordered [1]-[30] |
| `fix_references_final.py` | Final reference cleanup | ✅ References finalized |

### Metadata Generation (Not Tables)
| Script | Reason for Archiving | Purpose |
|--------|---------------------|---------|
| `create_detailed_metadata.py` | Metadata generation | Created metadata JSON files |
| `create_image_metadata.py` | Image metadata | Created image metadata |

---

## ⚠️ Important Notes

1. **Do NOT delete** these files - they may be needed for:
   - Historical reference
   - Understanding development process
   - Rollback if needed

2. **Do NOT use** these scripts - they are:
   - Obsolete
   - May have incorrect data
   - Replaced by better versions

3. **Active Scripts** are in:
   ```
   luaran/templates/code/
   ├── generate_table1_dataset_augmentation.py
   ├── generate_table2_detection_performance.py
   ├── generate_tables3456_classification.py
   ├── generate_table7_sota_comparison.py
   └── README.md
   ```

---

## 🔄 Migration Summary

**Before (Root Directory):**
- 27 Python scripts (messy!)
- Table scripts mixed with utilities
- Hard to find which script to use

**After (Organized):**
- **Root:** 9 active utility scripts only
- **`luaran/templates/code/`:** 4 FINAL table generation scripts
- **`_archived_scripts/`:** 20 obsolete scripts

**Result:** 70% cleaner root directory! ✨

---

## 📊 Final Table Scripts Location

**Use These (Active):**
```
luaran/templates/code/
├── generate_table1_dataset_augmentation.py       → Table 1
├── generate_table2_detection_performance.py      → Table 2
├── generate_tables3456_classification.py         → Tables 3-6
└── generate_table7_sota_comparison.py            → Table 7
```

**Archived (Obsolete):**
```
_archived_scripts/
├── create_table*.py           (old versions)
├── *citation*.py              (one-time use, done)
└── create_*_metadata.py       (metadata, not tables)
```

---

## 🎯 If You Need to Restore

If for any reason you need an archived script:
```bash
# Copy back to root (but DON'T use it!)
cp _archived_scripts/old_script.py ./

# Better: Check luaran/templates/code/ for active version
cd luaran/templates/code/
python generate_table1_dataset_augmentation.py
```

---

**Last Updated:** 2025-10-26
**Archive Reason:** Project cleanup and organization
**Status:** ✅ Safe to keep archived
