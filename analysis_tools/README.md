# Analysis Tools - Malaria Detection

Simple and organized tools for monitoring and analyzing the malaria detection experiments.

## 🚀 Quick Status Check

```bash
# Quick training status (concise output)
python analysis_tools/simple_monitor.py
```

## 📊 Available Tools

### Essential Tools

1. **`simple_monitor.py`** - Concise training progress monitor
   - Shows total models trained by category
   - Lists recently completed models (last 10 minutes)
   - Key process status without consuming context
   - Run: `python analysis_tools/simple_monitor.py`

2. **`consolidate_results.py`** - Results consolidation system
   - Scans and organizes all 105+ experiment results
   - Creates comprehensive summary files
   - Identifies best performing combinations
   - Run: `python analysis_tools/consolidate_results.py`

## 📁 Current Results Structure

```
results/current_experiments/training/
├── detection/
│   ├── yolo8_detection/
│   ├── yolo11_detection/
│   └── rtdetr_detection/
└── classification/
    ├── yolov8_classification/
    └── pytorch_classification/

consolidated_results/
├── RINGKASAN_LENGKAP.md          # Comprehensive summary
├── best_models_comparison.png    # Performance visualization
├── best_models/                  # Top model weights
└── summary_data.json            # Machine-readable results
```

## 🏆 Best Results (Quick Reference)

- **Champion**: Ground Truth → EfficientNet (100% accuracy)
- **Total Trained Models**: 105+
- **Categories**: Detection (2), Classification (25), Combinations (50+), Completed (7)

## 📈 Usage Tips

1. **For quick status**: Use `simple_monitor.py` - gives concise updates without long outputs
2. **For full analysis**: Check `consolidated_results/RINGKASAN_LENGKAP.md`
3. **For research**: Use `consolidate_results.py` to organize scattered results

## 🔧 Background Training

The system is running 50+ background training processes for all detection → classification combinations. Use the simple monitor to track progress without consuming excessive context.