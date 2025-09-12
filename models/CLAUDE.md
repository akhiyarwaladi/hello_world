# Models Directory - Context for Claude

## Purpose
Contains model weights and configurations for YOLOv8 and RT-DETR models.

## Structure
```
models/
├── yolov8/               # YOLOv8 model weights (GITIGNORED)
│   ├── .gitkeep         # Placeholder
│   └── *.pt files       # Model weights (ignored)
└── rtdetr/              # RT-DETR model weights (GITIGNORED)
    ├── .gitkeep         # Placeholder  
    └── *.pth files      # Model weights (ignored)
```

## Model Types
- **YOLOv8** - Primary object detection model for malaria parasite detection
- **RT-DETR** - Alternative detection transformer model for comparison

## Training Status (Updated: December 12, 2024)
1. ✅ **YOLOv8 Training ACTIVE** - Multiple training processes running on CPU
2. 🔄 **Data Pipeline FIXED** - Species mapping corrected for proper 6-class training
3. 📊 **Training Progress** - Real-time monitoring of multiple training sessions
4. 💾 **Model Weights** - Will be saved here upon training completion

## Current Training Sessions
- **yolo_classify** - Training on legacy data format 
- **yolo_classify_integrated** - Training on integrated dataset format
- **CPU-based training** - Using device=cpu due to GPU availability

## Important Notes
- All model weight files (*.pt, *.pth, *.onnx, etc.) are **GITIGNORED**
- Only configuration files and .gitkeep placeholders are tracked
- **TRAINING IN PROGRESS** - Multiple active YOLOv8 sessions