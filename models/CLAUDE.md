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

## Training Process
1. Models will be trained on processed malaria datasets
2. 6-class classification: P_falciparum, P_vivax, P_malariae, P_ovale, Mixed_infection, Uninfected  
3. Model weights saved here after training completion

## Important Notes
- All model weight files (*.pt, *.pth, *.onnx, etc.) are **GITIGNORED**
- Only configuration files and .gitkeep placeholders are tracked
- Model training will begin after pipeline completion