# Suggested Commands

## Setup Commands
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run setup script
bash setup.sh
```

## Data Pipeline Commands
```bash
# 1. Download datasets
python scripts/01_download_datasets.py --config config/dataset_config.yaml

# 2. Preprocess data  
python scripts/02_preprocess_data.py --resize 640

# 3. Integrate datasets
python scripts/03_integrate_datasets.py --output data/processed

# 4. Convert to YOLO format
python scripts/04_convert_to_yolo.py --format yolov8

# 5. Augment data
python scripts/05_augment_data.py --min-samples 500

# 6. Split dataset
python scripts/06_split_dataset.py --train 0.7 --val 0.15 --test 0.15
```

## Training Commands
```bash
# YOLOv8 training
yolo task=detect mode=train model=yolov8m.pt data=config/yolo_config.yaml epochs=100

# Custom training script
python scripts/train_model.py --model yolov8m --epochs 100 --batch 16
```

## System Commands
- `ls` - List files
- `cd` - Change directory
- `pwd` - Current directory
- `find` - Find files (better to use Serena tools)
- `grep` - Search text (better to use Serena tools)
- `git` - Version control
- `pip` - Package management