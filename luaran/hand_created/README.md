# Hand-Created Publication Content

## ✍️ EDIT FILES IN THIS DIRECTORY

This directory contains **manually created** research outputs. Unlike `auto_generated/`, you are **encouraged to edit** files here.

**Key Principle**: This is your workspace for papers, reports, and documentation that require human authorship and creativity.

---

## 📁 Directory Structure

```
hand_created/
├── papers/                  # Research manuscripts (Markdown)
│   ├── Draft_Journal_Q1_IEEE_TMI.md        # IEEE TMI submission draft
│   ├── JICEST_Paper.md                     # JICEST conference paper
│   ├── KINETIK_10_PAGES_NARRATIVE.md       # KINETIK journal paper
│   └── exports/             # DOCX/PDF exports for submission
│       ├── JICEST_Paper.docx
│       ├── JICEST_Paper.pdf
│       ├── KINETIK_10_PAGES.docx
│       └── ...
│
├── reports/                 # Progress reports
│   ├── Laporan_Kemajuan.md                 # Progress report (Markdown)
│   ├── Laporan Kemajuan Malaria.docx       # Progress report (DOCX)
│   └── Laporan Kemajuan Malaria.pdf        # Progress report (PDF)
│
├── documentation/           # Supporting documentation
│   ├── README.md                           # Documentation overview
│   ├── DATA_VERIFICATION_REPORT.md         # Data integrity verification
│   ├── VERIFIED_REFERENCES_40.md           # Bibliography verification
│   ├── FIGURE_ENHANCEMENT_SUMMARY.md       # Figure enhancement log
│   ├── REORGANISASI_REFERENSI_BERURUTAN.md # Reference reorganization
│   └── SUMMARY_REPORT_FOR_USER.md          # User-facing summary
│
└── README.md               # This file
```

---

## 📝 What Should Be Manually Created?

### ✅ Always Manual (Hand-Created)

1. **Research Papers** - Manuscripts for journals/conferences
2. **Reports** - Progress reports, final reports
3. **Documentation** - READMEs, verification reports, summaries
4. **Literature Review** - Related work, bibliography
5. **Discussion & Conclusions** - Interpretation and implications
6. **Methodology Narrative** - Detailed method descriptions

### ⚙️ Always Auto-Generated (See `../auto_generated/`)

1. **Figures** - Pipeline diagrams, augmentation visualizations, performance plots
2. **Tables** - Classification results, detection metrics, dataset statistics
3. **Metadata** - Generation timestamps, integrity checksums

### 🤔 Hybrid (Manual + Auto)

1. **Results Section** - Manual narrative + auto-generated tables/figures
2. **Experimental Setup** - Manual description + auto-generated statistics
3. **Performance Analysis** - Manual interpretation + auto-generated plots

---

## 🚀 Quick Start Workflows

### Workflow 1: Write a New Paper

```bash
# Step 1: Copy template (if needed)
cp ../templates/Template\ Kinetik\ Mendeley.docx papers/New_Paper.md

# Step 2: Write content in Markdown
# Edit papers/New_Paper.md
# Reference auto_generated data using relative paths

# Step 3: Export to DOCX for submission
pandoc papers/New_Paper.md -o papers/exports/New_Paper.docx

# Step 4: Generate PDF (via Word or Pandoc)
# Option A: Open DOCX in Word → Save as PDF
# Option B: Direct PDF export
pandoc papers/New_Paper.md -o papers/exports/New_Paper.pdf
```

### Workflow 2: Update Existing Paper with Latest Results

```bash
# Step 1: Ensure latest auto_generated data
python scripts/publication/generate_all_publication_outputs.py

# Step 2: Update paper references (if needed)
# Edit papers/JICEST_Paper.md
# Update figure/table paths or captions

# Step 3: Re-export to DOCX
pandoc papers/JICEST_Paper.md -o papers/exports/JICEST_Paper.docx

# Step 4: Verify data integrity
python scripts/publication/verify_publication_data.py
```

### Workflow 3: Create Progress Report

```bash
# Step 1: Write report in Markdown
# Edit reports/Laporan_Kemajuan.md

# Step 2: Export to DOCX (with template)
pandoc reports/Laporan_Kemajuan.md \
  --reference-doc=../templates/template_laporan_kemajuan.docx \
  -o reports/Laporan\ Kemajuan\ Malaria.docx

# Step 3: Generate PDF from DOCX
# Open in Word → Save as PDF
# Or use LibreOffice:
libreoffice --headless --convert-to pdf reports/Laporan\ Kemajuan\ Malaria.docx
```

---

## 📄 Paper Organization (papers/)

### Current Papers

| Paper | Status | Target Venue | Format |
|-------|--------|--------------|--------|
| `Draft_Journal_Q1_IEEE_TMI.md` | Draft | IEEE TMI (Q1) | Markdown |
| `JICEST_Paper.md` | Complete | JICEST Conference | Markdown |
| `KINETIK_10_PAGES_NARRATIVE.md` | Complete | KINETIK Journal | Markdown |

### File Naming Convention

**Source files** (Markdown):
- `[Venue]_Paper.md` - Example: `JICEST_Paper.md`
- `Draft_[Venue]_[JournalName].md` - Example: `Draft_Journal_Q1_IEEE_TMI.md`

**Exported files** (DOCX/PDF):
- `[Venue]_Paper.docx` - Example: `JICEST_Paper.docx`
- `[Venue]_Paper.pdf` - Example: `JICEST_Paper.pdf`
- Store in `exports/` subdirectory

### Best Practices

1. **Write in Markdown** - Better version control, easier collaboration
2. **Use clear section headers** - `## Introduction`, `## Methods`, `## Results`
3. **Reference auto_generated data** - Use relative paths: `../../auto_generated/figures/`
4. **Keep source and exports** - Don't delete `.md` after exporting
5. **Version control exports** - Include DOCX/PDF in Git for submission tracking

---

## 📊 Referencing Auto-Generated Data

### Figures

**In Markdown**:
```markdown
<!-- Pipeline Architecture -->
![Figure 1: Pipeline Architecture](../../auto_generated/figures/pipeline_diagrams/pipeline_architecture_publication.png)

<!-- Augmentation Visualization -->
![Figure 2: Data Augmentation - IML Lifecycle](../../auto_generated/figures/augmentation/augmentation_iml_lifecycle_combined.png)

<!-- Performance Analysis -->
![Figure 3: Confusion Matrix - YOLO11](../../auto_generated/figures/performance/confusion_matrix_yolo11.png)
```

**With captions and labels**:
```markdown
<figure id="fig:pipeline">
  <img src="../../auto_generated/figures/pipeline_diagrams/pipeline_architecture_publication.png"
       alt="Pipeline Architecture"
       width="100%"/>
  <figcaption>Figure 1: Proposed malaria detection pipeline architecture showing shared classification approach.</figcaption>
</figure>
```

### Tables

**Option 1: Markdown tables** (copy data from auto_generated CSV):
```markdown
**Table 1**: Detection performance across datasets

| Dataset | Model | mAP@50 | mAP@50-95 | Precision | Recall |
|---------|-------|--------|-----------|-----------|--------|
| IML Lifecycle | YOLO11 | 0.9457 | 0.8006 | 0.9160 | 0.9510 |
| MP-IDB Species | YOLO11 | 0.9288 | 0.5575 | 0.8868 | 0.8957 |
| MP-IDB Stages | YOLO11 | 0.9335 | 0.5612 | 0.8912 | 0.9023 |

_Source: `../../auto_generated/tables/detection/detection_performance_all_datasets.csv`_
```

**Option 2: CSV include** (using Pandoc filter):
```markdown
<!-- Include table from CSV -->
```{.table #tbl:detection source="../../auto_generated/tables/detection/detection_performance_all_datasets.csv"}
Table 1: Detection performance across datasets
```
```

**Option 3: Reference in text**:
```markdown
Detection results are shown in **Table 1** (see `../../auto_generated/tables/detection/detection_performance_all_datasets.csv` for detailed metrics).
```

### Data Verification

**Always cite source data**:
```markdown
Classification results (Table 9) show Focal Loss outperforms Class-Balanced Loss across all datasets, with EfficientNet-B1 achieving 80.90% accuracy on IML Lifecycle (source: `../../auto_generated/tables/classification/table9_focal_loss.csv`, verified 2025-10-12).
```

---

## 📤 Export Workflows

### Markdown → DOCX (Pandoc)

**Basic export**:
```bash
pandoc papers/JICEST_Paper.md -o papers/exports/JICEST_Paper.docx
```

**With template** (preserves journal formatting):
```bash
pandoc papers/JICEST_Paper.md \
  --reference-doc=../templates/Template\ Kinetik\ Mendeley.docx \
  -o papers/exports/JICEST_Paper.docx
```

**With bibliography**:
```bash
pandoc papers/JICEST_Paper.md \
  --bibliography=documentation/VERIFIED_REFERENCES_40.md \
  --csl=ieee.csl \
  -o papers/exports/JICEST_Paper.docx
```

**With resource paths** (for figures):
```bash
pandoc papers/JICEST_Paper.md \
  --resource-path=.:../../auto_generated/figures:../../auto_generated/tables \
  --reference-doc=../templates/Template\ Kinetik\ Mendeley.docx \
  -o papers/exports/JICEST_Paper.docx
```

### Markdown → PDF (Pandoc)

**Direct PDF** (requires LaTeX):
```bash
pandoc papers/JICEST_Paper.md \
  --pdf-engine=xelatex \
  -o papers/exports/JICEST_Paper.pdf
```

**Via DOCX** (recommended):
```bash
# Step 1: Generate DOCX
pandoc papers/JICEST_Paper.md -o papers/exports/JICEST_Paper.docx

# Step 2: Open in Word → Save as PDF
# Or use LibreOffice:
libreoffice --headless --convert-to pdf papers/exports/JICEST_Paper.docx \
  --outdir papers/exports/
```

### DOCX → PDF (Without Pandoc)

**Using Microsoft Word**:
1. Open `papers/exports/JICEST_Paper.docx`
2. File → Save As → PDF
3. Save to `papers/exports/JICEST_Paper.pdf`

**Using LibreOffice** (command line):
```bash
libreoffice --headless --convert-to pdf \
  papers/exports/JICEST_Paper.docx \
  --outdir papers/exports/
```

**Using Google Docs**:
1. Upload DOCX to Google Drive
2. Open with Google Docs
3. File → Download → PDF

---

## 📋 Report Organization (reports/)

### Current Reports

| Report | Language | Format | Purpose |
|--------|----------|--------|---------|
| `Laporan_Kemajuan.md` | Bahasa Indonesia | Markdown | Progress report (source) |
| `Laporan Kemajuan Malaria.docx` | Bahasa Indonesia | DOCX | Progress report (export) |
| `Laporan Kemajuan Malaria.pdf` | Bahasa Indonesia | PDF | Progress report (final) |

### Report Structure

**Standard sections**:
1. **Pendahuluan** (Introduction)
2. **Tinjauan Pustaka** (Literature Review)
3. **Metodologi Penelitian** (Methodology)
4. **Hasil dan Pembahasan** (Results and Discussion)
5. **Kesimpulan** (Conclusions)
6. **Daftar Pustaka** (References)

### Report Best Practices

1. **Use official template** - Copy from `../templates/template_laporan_kemajuan.docx`
2. **Include all experiments** - Reference latest auto_generated data
3. **Bilingual captions** - Provide both ID and EN for figures/tables
4. **Consistent formatting** - Follow template styles
5. **Export to PDF** - Final submission in PDF format

---

## 📚 Documentation Organization (documentation/)

### Current Documentation

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | Documentation overview | All users |
| `DATA_VERIFICATION_REPORT.md` | Data integrity verification | Researchers |
| `VERIFIED_REFERENCES_40.md` | Bibliography verification | Paper authors |
| `FIGURE_ENHANCEMENT_SUMMARY.md` | Figure enhancement log | Technical team |
| `REORGANISASI_REFERENSI_BERURUTAN.md` | Reference reorganization | Paper authors |
| `SUMMARY_REPORT_FOR_USER.md` | User-facing summary | Non-technical users |

### Documentation Types

1. **Technical Reports** - Data verification, system analysis
2. **User Guides** - How-to documents, workflows
3. **Summaries** - Executive summaries, quick references
4. **Verification** - Quality assurance, integrity checks

### Documentation Best Practices

1. **Clear titles** - Use descriptive filenames
2. **Consistent format** - Follow Markdown best practices
3. **Update regularly** - Keep documentation current
4. **Cross-reference** - Link to related documents
5. **Version history** - Track major changes

---

## ✅ Quality Checklist

### Before Paper Submission

- [ ] Latest auto_generated data referenced (regenerate if needed)
- [ ] All figures and tables correctly cited
- [ ] Data sources verified (`verify_publication_data.py`)
- [ ] Exported to required format (DOCX/PDF)
- [ ] Template formatting preserved
- [ ] Bibliography complete and formatted
- [ ] No broken image links
- [ ] Figures have captions and labels
- [ ] Tables have titles and sources
- [ ] Acknowledgments included

### Before Report Submission

- [ ] All sections complete (Pendahuluan → Kesimpulan)
- [ ] Latest experiment results included
- [ ] Figures in correct language (Bahasa Indonesia)
- [ ] Template formatting preserved
- [ ] Exported to DOCX and PDF
- [ ] Page numbers correct
- [ ] Table of contents updated
- [ ] References formatted correctly

### Before Documentation Update

- [ ] Content accurate and up-to-date
- [ ] Cross-references valid
- [ ] Examples tested and working
- [ ] Markdown syntax correct
- [ ] Links not broken
- [ ] Version history updated

---

## 🔧 Common Export Issues & Solutions

### Issue 1: Figures not appearing in DOCX

**Cause**: Incorrect relative paths

**Solution**:
```bash
# Use --resource-path flag
pandoc papers/JICEST_Paper.md \
  --resource-path=.:../../auto_generated/figures \
  -o papers/exports/JICEST_Paper.docx
```

### Issue 2: Table formatting broken in DOCX

**Cause**: Pandoc table conversion issues

**Solution**:
```bash
# Use grid tables (better compatibility)
+-----------+--------+--------+
| Dataset   | mAP@50 | Recall |
+===========+========+========+
| IML       | 0.9457 | 0.9510 |
+-----------+--------+--------+

# Or use reference DOCX template
pandoc papers/JICEST_Paper.md \
  --reference-doc=../templates/Template\ Kinetik\ Mendeley.docx \
  -o papers/exports/JICEST_Paper.docx
```

### Issue 3: Bibliography not included

**Cause**: Missing bibliography flag

**Solution**:
```bash
pandoc papers/JICEST_Paper.md \
  --bibliography=documentation/VERIFIED_REFERENCES_40.md \
  --csl=ieee.csl \
  -o papers/exports/JICEST_Paper.docx
```

### Issue 4: PDF export fails

**Cause**: Missing LaTeX installation

**Solution**:
```bash
# Option A: Install LaTeX (MiKTeX or TeX Live)
# Then use:
pandoc papers/JICEST_Paper.md --pdf-engine=xelatex -o papers/exports/JICEST_Paper.pdf

# Option B: Export via DOCX (recommended)
pandoc papers/JICEST_Paper.md -o papers/exports/JICEST_Paper.docx
# Then: Open in Word → Save as PDF
```

### Issue 5: Template formatting lost

**Cause**: Not using reference DOCX

**Solution**:
```bash
# Always use --reference-doc for template preservation
pandoc papers/JICEST_Paper.md \
  --reference-doc=../templates/Template\ Kinetik\ Mendeley.docx \
  -o papers/exports/JICEST_Paper.docx
```

---

## 🎨 Writing Best Practices

### Markdown Style Guide

**Headers**:
```markdown
# Main Title (H1) - Use only once
## Section (H2) - Major sections
### Subsection (H3) - Subsections
#### Subsubsection (H4) - Details
```

**Emphasis**:
```markdown
*italic* or _italic_
**bold** or __bold__
***bold italic***
`code` or ``code``
```

**Lists**:
```markdown
<!-- Unordered -->
- Item 1
- Item 2
  - Nested item

<!-- Ordered -->
1. First
2. Second
3. Third
```

**Links & References**:
```markdown
[Link text](URL)
[Internal reference](#section-id)
[Figure reference](#fig:pipeline)
![Image](path/to/image.png)
```

### Citation Style

**IEEE Style** (for engineering papers):
```markdown
According to recent studies [1], malaria detection using deep learning shows promising results [2], [3].

## References
[1] Author, "Title," Journal, vol. X, no. Y, pp. Z, Year.
[2] ...
```

**APA Style** (for medical papers):
```markdown
Recent studies (Author, Year) show that deep learning improves malaria detection (Author1 & Author2, Year).

## References
Author, A. B. (Year). Title. Journal, Volume(Issue), pages.
```

### Figure & Table Captions

**Figures**:
```markdown
![Figure 1: Pipeline architecture showing detection and classification stages](../../auto_generated/figures/pipeline_diagrams/pipeline_architecture_publication.png)

**Figure 1**: Pipeline architecture showing detection and classification stages. The shared classification approach reduces storage by ~70% and training time by ~60%.
```

**Tables**:
```markdown
**Table 1**: Detection performance across datasets

| Dataset | Model | mAP@50 | mAP@50-95 |
|---------|-------|--------|-----------|
| ... | ... | ... | ... |

_Source: Auto-generated from experiment optA_20251012_103000_
```

---

## 🔗 Integration with Auto-Generated Data

### Data Flow

```
Pipeline Experiments (results/optA_*/)
           ↓
Auto-Generated Outputs (../auto_generated/)
           ↓
Referenced in Hand-Created Papers (papers/*.md)
           ↓
Exported to DOCX/PDF (papers/exports/)
           ↓
Submitted to Journals/Conferences
```

### Verification Workflow

```bash
# Step 1: Ensure latest auto_generated data
python scripts/publication/generate_all_publication_outputs.py

# Step 2: Verify data integrity
python scripts/publication/verify_publication_data.py

# Step 3: Update paper references (if needed)
# Edit papers/JICEST_Paper.md

# Step 4: Export to submission format
pandoc papers/JICEST_Paper.md -o papers/exports/JICEST_Paper.docx

# Step 5: Final verification
python scripts/publication/verify_publication_data.py --check-citations
```

---

## 📚 Additional Resources

- **Main Project Documentation**: See `../../CLAUDE.md` for full pipeline guide
- **Luaran Overview**: See `../README.md` for directory structure
- **Auto-Generated Guide**: See `../auto_generated/README.md` for regeneration instructions
- **Templates**: See `../templates/` for official templates
- **Pandoc Manual**: https://pandoc.org/MANUAL.html
- **Markdown Guide**: https://www.markdownguide.org/

---

## 🔄 Change Log

### 2025-10-12: Initial Hand-Created Structure
- ✅ Organized papers, reports, documentation
- ✅ Created export workflows
- ✅ Integrated with auto_generated data
- ✅ Added quality checklists

---

*Last Updated: 2025-10-12*
*Maintained by: Research Team*
*✍️ EDIT FILES IN THIS DIRECTORY AS NEEDED*
