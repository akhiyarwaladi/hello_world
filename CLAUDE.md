# Malaria Detection Project - Context for Claude

## 🎯 Project Overview
Comprehensive malaria detection system using YOLOv8, YOLOv11, and RT-DETR models for microscopy image analysis.

**Current Status: ✅ PRODUCTION READY - Full Pipeline Implemented with Organized Structure**

## 🚀 Pipeline Architecture (COMPLETED)

### Unified Interface
- **Main Pipeline**: `pipeline.py` - Unified interface for all operations
- **Auto Organization**: Results automatically organized in structured folders
- **Background Training**: Multiple parallel training processes supported
- **Full Automation**: Complete pipeline from data download to model training

### Complete Data Flow
```
1. Download (01_) → 2. Preprocess (02_) → 3. Integrate (03_) →
4. Convert (04_) → 5. Augment (05_) → 6. Split (06_) → 7. Train
```

## 📁 Organized Results Structure (AUTO-GENERATED)

```
results/
├── current_experiments/    # Active training/validation
│   ├── validation/        # Quick tests & validation
│   ├── training/          # Full training experiments
│   └── comparison/        # Model comparisons
├── completed_models/      # Production-ready models
├── publications/         # Publication-ready exports
├── archive/              # Historical experiments
└── experiment_logs/      # All experiment logs
```

## 🎯 Available Models & Datasets

### Models (5 total)
- `yolov8_detection` - YOLOv8 parasite detection
- `yolov11_detection` - YOLOv11 parasite detection
- `rtdetr_detection` - RT-DETR parasite detection
- `yolov8_classification` - YOLOv8 species classification
- `yolov11_classification` - YOLOv11 species classification

### Datasets (4 total)
- `detection_multispecies` - 4-class detection (falciparum, malariae, ovale, vivax)
- `detection_fixed` - Single-class detection (parasite)
- `classification_multispecies` - 4-class classification
- `classification_crops` - Single-class classification

## ⚡ Quick Commands

### Pipeline Status & Control
```bash
python pipeline.py status                    # System status
python pipeline.py list                      # List models/datasets
python pipeline.py validate --models all     # Quick validation
```

### Training (Background Supported)
```bash
python pipeline.py train yolov8_detection --epochs 50 --background
python pipeline.py train yolov8_classification --epochs 50 --background
python pipeline.py train yolov11_detection --epochs 30 --background
```

### Full Pipeline from Scratch
```bash
# Option 1: Automated
python scripts/run_full_pipeline.py

# Option 2: Step by step
python scripts/01_download_datasets.py
python scripts/02_preprocess_data.py
# ... continue with 03, 04, 05, 06

# Option 3: Unified pipeline
python pipeline.py validate --models all
```

## 🔧 Technical Implementation

### Organized Scripts Structure (✅ CLEANED)
```
scripts/
├── 01_download_datasets.py      # Data download from all sources
├── 02_preprocess_data.py         # Image preprocessing & quality checks
├── 03_integrate_datasets.py      # Dataset integration & species mapping
├── 04_convert_to_yolo.py         # YOLO format conversion
├── 05_augment_data.py            # Data augmentation for class balance
├── 06_split_dataset.py           # Train/val/test splitting
├── 07_train_yolo_detection.py    # YOLOv8 detection training
├── 08_train_yolo11_detection.py  # YOLOv11 detection training
├── 09_train_rtdetr_detection.py  # RT-DETR detection training
├── 10_train_yolo_detection.py    # Legacy detection training (DEPRECATED)
├── 11_train_classification_crops.py  # Classification training
└── utils/                        # Helper utilities
    ├── results_manager.py        # Automatic folder organization
    ├── dataset_utils.py          # Dataset helpers
    ├── image_utils.py            # Image processing utilities
    └── annotation_utils.py       # Annotation conversion tools
```

### Key Files
- `pipeline.py` (437 lines) - Main unified interface
- `utils/results_manager.py` - Automatic folder organization
- `config/models.yaml` - Model configurations (updated with numbered scripts)
- `config/datasets.yaml` - Dataset configurations
- `config/results_structure.yaml` - Folder structure definition

### Background Processes
- Multiple training processes can run in parallel
- Automatic progress monitoring and logging
- Organized output in structured folders
- Resume capability for interrupted training

## 🎯 Current Training Status
- **Multiple Active Processes**: 10+ background training jobs running
- **Organized Structure**: All results automatically organized
- **No Manual Intervention**: Scripts automatically save to correct folders
- **Parallel Processing**: Detection + Classification training simultaneously

## 💾 Data Management

### Data Download Support
- **Complete automation**: Download all 6 datasets automatically
- **Resume capability**: Interrupted downloads resume automatically
- **Data validation**: Checksum verification for data integrity
- **Space requirement**: ~15GB for all datasets

### Safe Re-running
```bash
# Safe to delete and re-download
rm -rf data/raw data/processed data/integrated
python scripts/01_download_datasets.py
```

## 🔍 Monitoring & Results

### Real-time Monitoring
```bash
watch -n 30 'python pipeline.py status'
ps aux | grep python  # Check background processes
```

### Results Export
```bash
python pipeline.py evaluate --comprehensive  # Full evaluation
python pipeline.py export --format journal   # Publication export
```

## 🚨 Important Notes for Claude

### Pipeline Usage
- **Always check status first**: `python pipeline.py status`
- **Use background training**: Add `--background` for long training
- **Monitor organization**: Results auto-organize, no manual intervention needed
- **Full automation available**: `python scripts/run_full_pipeline.py` does everything

### Common Workflows
1. **Development**: `validate → quick train → evaluate`
2. **Production**: `full pipeline → parallel training → comprehensive evaluation`
3. **Research**: `compare models → export results → publish`

### Error Handling
- **Import errors**: All training scripts updated with correct import paths
- **Memory issues**: Reduce batch size (`--batch 4` for CPU)
- **Download issues**: Safe to delete and re-download data
- **Organization**: Folders auto-create, no manual setup needed

## 📊 Performance & Optimization

### CPU Training (Current Setup)
- Optimal batch sizes: detection=4-8, classification=2-4
- NNPACK disabled for stability
- Background processing for long runs

### Results Tracking
- All training automatically logged
- CSV results generated per experiment
- Organized by model type and experiment name
- Publication-ready export available

---

## 🎉 Status: FULLY OPERATIONAL

✅ **Complete pipeline implemented**
✅ **Organized structure working**
✅ **Multiple training processes active**
✅ **Automatic result organization**
✅ **Background processing functional**
✅ **Full documentation available**

**Ready for production use with minimal setup required.**