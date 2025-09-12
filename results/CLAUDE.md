# Results Directory - Context for Claude

## Purpose
Contains training results, logs, and predictions from model experiments.

## Structure  
```
results/
├── weights/              # Trained model checkpoints (GITIGNORED)
├── logs/                 # Training logs and metrics (GITIGNORED)
├── predictions/          # Model predictions and visualizations (GITIGNORED)
└── analysis/             # Performance analysis and reports
```

## Contents (After Training)
- **Weights** - Best model checkpoints, validation metrics
- **Logs** - Training curves, loss plots, validation scores
- **Predictions** - Sample predictions, confusion matrices, detection visualizations
- **Analysis** - Performance comparison between YOLOv8 and RT-DETR

## Metrics Tracked
- Detection accuracy per species
- Precision, Recall, F1-score for each class
- mAP (mean Average Precision) 
- Training/validation loss curves
- Inference speed benchmarks

## Important Notes
- Large result files are **GITIGNORED** to prevent repo bloat
- Analysis reports and summaries will be committed for reference
- Results generated after successful model training completion