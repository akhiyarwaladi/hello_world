# 📊 Unified Visualization System

**Centralized, modular, extensible visualization pipeline for malaria detection research.**

---

## 🎯 **OVERVIEW**

This system provides a unified, production-ready framework for generating and analyzing visualizations across all experiments.

### **Key Features:**
✅ **Centralized** - One entry point for all visualization tasks
✅ **Modular** - Clean separation of concerns (selectors, reporters, generators)
✅ **Extensible** - Easy to add new visualization types, selectors, or report formats
✅ **Production-Ready** - Error handling, logging, auto-detection
✅ **Well-Documented** - Clear usage examples and extension guides

---

## 🚀 **QUICK START**

### **Basic Usage:**

```bash
# Process latest experiment (auto-detect)
python scripts/visualization/generate_all_test_visualizations.py

# Specific experiment
python scripts/visualization/generate_all_test_visualizations.py \
  --experiment-dir results/optA_20251207_233941

# Custom output directory
python scripts/visualization/generate_all_test_visualizations.py \
  --output custom_analysis \
  --top-n 10

# Skip training curves (faster)
python scripts/visualization/generate_all_test_visualizations.py \
  --skip-training-curves
```

### **Expected Output:**

```
results/optA_20251207_233941/
└── visualization_summary/
    ├── selected_detection_errors.csv           # Detection error cases
    ├── selected_classification_errors.csv      # Classification error cases
    ├── selected_error_images.csv               # Combined (old format)
    ├── visualization_report.md                 # Human-readable report
    ├── visualization_metadata.json             # Machine-readable metadata
    └── training_curves/                        # Training curve figures
        ├── accuracy_iml_lifecycle.png
        ├── accuracy_mp_idb_species.png
        ├── accuracy_mp_idb_stages.png
        └── accuracy_md_2019_stages.png
```

---

## 🏗️ **ARCHITECTURE**

```
scripts/visualization/
│
├── generate_all_test_visualizations.py        # 🎯 MAIN ORCHESTRATOR (start here!)
│
├── [Visualization Generators]
│   ├── generate_detection_only_with_metadata.py        ✅ Existing
│   ├── generate_classification_only_with_metadata.py   ✅ Existing
│   └── generate_training_curves.py                     🆕 Refactored & modular
│
├── [Selectors - Error Case Selection]
│   ├── selectors/
│   │   ├── base_selector.py                   # Abstract base class
│   │   ├── detection_error_selector.py        # Detection errors (FP, FN, Mixed)
│   │   └── classification_error_selector.py   # Classification errors
│
├── [Reporters - Multi-Format Output]
│   ├── reporters/
│   │   ├── base_reporter.py                   # Abstract base class
│   │   ├── csv_reporter.py                    # CSV export
│   │   ├── markdown_reporter.py               # Human-readable reports
│   │   └── json_reporter.py                   # Machine-readable metadata
│
└── README_VISUALIZATION.md                     # This file
```

### **Design Principles:**
1. **Modularity** - Each component has single responsibility
2. **Extensibility** - Plugin architecture for easy additions
3. **Reusability** - Reuse existing production code
4. **Maintainability** - Clear interfaces, good documentation
5. **Testability** - Each module can be tested independently

---

## 📚 **DETAILED USAGE**

### **1. Main Orchestrator**

```python
from scripts.visualization.generate_all_test_visualizations import VisualizationOrchestrator

# Create orchestrator
orchestrator = VisualizationOrchestrator(
    experiment_root='results/optA_20251207_233941',
    output_dir='custom_analysis',
    top_n=5
)

# Run full pipeline
orchestrator.run_full_pipeline()
```

### **2. Detection Error Selector (Standalone)**

```python
from scripts.visualization.selectors import DetectionErrorSelector

# Create selector
selector = DetectionErrorSelector(top_n=5, include_perfect=True)

# Select from CSV
selected = selector.select_from_csv('detection_metadata.csv')

# Get statistics
stats = selector.get_category_stats(full_df)
selector.print_summary()

# Save results
selector.save_results('selected.csv', selected)
```

### **3. Classification Error Selector (Standalone)**

```python
from scripts.visualization.selectors import ClassificationErrorSelector

# Create selector
selector = ClassificationErrorSelector(
    top_n=5,
    conf_threshold=0.75,
    include_perfect=True
)

# Select from multiple CSVs
csv_files = list(Path('visualizations').glob('*/classification_metadata_images.csv'))
selected = selector.select_from_multiple_csvs(csv_files)

# Analyze confusion patterns
confusion = selector.get_confusion_patterns('classification_metadata_boxes.csv')
```

### **4. Training Curves Generator (Standalone)**

```python
from scripts.visualization.generate_training_curves import TrainingCurvesGenerator

# Create generator
generator = TrainingCurvesGenerator(
    experiment_dir='results/optA_20251207_233941/experiments',
    output_dir='training_curves',
    dpi=400
)

# Generate all curves
results = generator.generate_all(plot_types=['accuracy', 'loss'])
```

### **5. Reporters (Standalone)**

```python
from scripts.visualization.reporters import CSVReporter, MarkdownReporter, JSONReporter

# CSV export
csv_reporter = CSVReporter()
csv_reporter.generate(dataframe, 'output.csv')

# Markdown report
md_reporter = MarkdownReporter(title="Analysis Report")
md_reporter.generate(analysis_data, 'report.md')

# JSON metadata
json_reporter = JSONReporter()
json_reporter.generate(metadata_dict, 'metadata.json')
```

---

## 🔧 **EXTENDING THE SYSTEM**

### **Adding a New Selector**

```python
# selectors/custom_selector.py
from .base_selector import BaseSelector

class MyCustomSelector(BaseSelector):
    """Select images based on custom criteria."""

    def select(self, metadata_df):
        # Your custom selection logic
        filtered = metadata_df[metadata_df['custom_metric'] > 0.9]
        return filtered.head(self.top_n)

    def get_selection_criteria(self):
        return {
            'type': 'custom',
            'criteria': 'high custom metric'
        }
```

Then register in `selectors/__init__.py`:

```python
from .custom_selector import MyCustomSelector
__all__ = [..., 'MyCustomSelector']
```

### **Adding a New Reporter**

```python
# reporters/latex_reporter.py
from .base_reporter import BaseReporter

class LaTeXReporter(BaseReporter):
    """Generate LaTeX tables for publication."""

    def generate(self, data, output_path):
        # Your LaTeX generation logic
        with open(output_path, 'w') as f:
            f.write("\\begin{table}\n")
            # ...
            f.write("\\end{table}\n")
        return True
```

### **Adding a New Visualization Type**

1. Create generator script: `generate_my_viz_with_metadata.py`
2. Follow existing pattern (detect model, generate viz, export CSV)
3. Add to orchestrator in `generate_all_test_visualizations.py`:

```python
def generate_my_visualization(self, experiments):
    """Generate my custom visualization."""
    for exp in experiments:
        # Call your generator
        pass
```

---

## 📊 **ERROR CATEGORIES**

### **Detection Errors:**
- **False Positives (FP)** - Model detects parasites that don't exist
- **False Negatives (FN)** - Model misses real parasites
- **Mixed (FP + FN)** - Both types of errors in same image
- **Perfect** - Correct detection (for comparison)

### **Classification Errors:**
- **All Wrong** - Complete classification failure
- **High Confidence Errors** - Wrong but confident (confidence > 0.75)
- **Mixed Results** - Some correct, some wrong
- **Low Confidence Correct** - Uncertain but right (confidence < 0.6)
- **Perfect** - All correct (for comparison)

---

## 📈 **SELECTION CRITERIA**

### **Paper Score System:**
- **9-10** - Best for paper (perfect with high confidence)
- **7-8** - Good for paper (correct or mostly correct)
- **5-6** - Challenging cases (mixed errors, interesting failures)
- **3-4** - Error cases (FP, FN, all wrong)

### **Selection Strategy:**
- Select top N cases per category per dataset per model
- Sort by relevance (errors → confidence → count)
- Include perfect cases for comparison
- Prioritize interesting failure modes

---

## 🔍 **TROUBLESHOOTING**

### **No experiments found:**
```bash
# Check experiment structure
ls results/optA_*/experiments/

# Specify explicit path
python generate_all_test_visualizations.py \
  --experiment-dir results/optA_20251207_233941
```

### **Missing metadata CSVs:**
```bash
# Regenerate visualizations first
python scripts/pipeline/generate_visualizations_with_metadata.py \
  --experiment-dir results/optA_20251207_233941/experiments/experiment_iml_lifecycle \
  --dataset-name iml_lifecycle \
  --num-classes 4
```

### **Training curves fail:**
```bash
# Skip training curves
python generate_all_test_visualizations.py --skip-training-curves

# Or generate separately
python scripts/visualization/generate_training_curves.py \
  --experiment-dir results/optA_20251207_233941/experiments \
  --output-dir training_curves
```

---

## 💡 **BEST PRACTICES**

1. **Always run full pipeline first** - Ensures all metadata exists
2. **Use top-n=5** for paper figures - Good balance of coverage and quality
3. **Include perfect cases** - Shows model capabilities
4. **Review Markdown report first** - Human-readable overview
5. **Use CSV for filtering** - Open in Excel for custom selection
6. **Keep JSON metadata** - Good for archival and reproducibility

---

## 📝 **CHANGELOG**

### **v1.0 (2025-12-10)**
- ✅ Initial modular architecture
- ✅ Detection error selector
- ✅ Classification error selector
- ✅ Training curves generator (refactored)
- ✅ Multi-format reporters (CSV, Markdown, JSON)
- ✅ Main orchestrator with auto-detection
- ✅ Comprehensive documentation

---

## 🎓 **FOR DEVELOPERS**

### **Code Style:**
- Follow PEP 8
- Use type hints
- Add docstrings (Google style)
- Write clear comments for complex logic

### **Testing:**
```bash
# Test selector
python -m scripts.visualization.selectors.detection_error_selector

# Test reporter
python -m scripts.visualization.reporters.csv_reporter

# Test full pipeline
python scripts/visualization/generate_all_test_visualizations.py --top-n 1
```

### **Adding to Pipeline:**
1. Create modular component (inherit base class)
2. Test standalone first
3. Add to orchestrator
4. Update documentation
5. Test end-to-end

---

**Maintainer:** Claude Sonnet 4.5
**Last Updated:** 2025-12-10
**Status:** Production Ready ✅
