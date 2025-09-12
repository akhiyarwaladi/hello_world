#!/bin/bash
# setup.sh - Setup script for Malaria YOLO Detection Repository

echo "🔬 Malaria Detection Setup Script"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,cache}
mkdir -p data/raw/{nih_cell,mp_idb,bbbc041,plasmoID,iml,uganda}
mkdir -p data/processed/{images/{train,val,test},labels/{train,val,test}}
mkdir -p models/{yolov8,rtdetr}
mkdir -p results/{weights,logs,predictions,analysis}
mkdir -p config
mkdir -p notebooks
mkdir -p tests

# Create configuration files
echo "⚙️ Creating configuration files..."

# Create dataset_config.yaml
cat > config/dataset_config.yaml << 'EOF'
datasets:
  nih_cell:
    url: "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
    type: "segmented_cells"
    classes: ["infected", "uninfected"]
    
  mp_idb:
    url: "https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis.git"
    type: "whole_slide"
    classes: ["P_falciparum", "P_vivax", "P_malariae", "P_ovale"]
    
  kaggle_nih:
    dataset: "iarunava/cell-images-for-detecting-malaria"
    type: "segmented_cells"
    classes: ["parasitized", "uninfected"]

augmentation:
  minority_threshold: 500
  techniques:
    - rotation: [-30, 30]
    - flip: ["horizontal", "vertical"]
    - brightness: [0.8, 1.2]
    - contrast: [0.8, 1.2]
EOF

# Create class_names.yaml
cat > config/class_names.yaml << 'EOF'
classes:
  0: "P_falciparum"
  1: "P_vivax"
  2: "P_malariae"
  3: "P_ovale"
  4: "Mixed_infection"
  5: "Uninfected"

colors:  # BGR format
  0: [255, 0, 0]       # Blue
  1: [0, 255, 0]       # Green
  2: [0, 0, 255]       # Red
  3: [255, 255, 0]     # Cyan
  4: [255, 0, 255]     # Magenta
  5: [128, 128, 128]   # Gray
EOF

# Setup Kaggle API (if credentials exist)
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "✓ Kaggle API credentials found"
    chmod 600 ~/.kaggle/kaggle.json
else
    echo "ℹ️ Kaggle API credentials not found"
    echo "   To use Kaggle datasets, place your kaggle.json in ~/.kaggle/"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Download datasets: python scripts/01_download_datasets.py"
echo "3. Process data: python scripts/02_preprocess_data.py"
echo "4. Train model: python scripts/train_yolo.py --model yolov8m --epochs 100"
echo ""
echo "For more information, see README.md"

# Create .gitignore file
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# PyCharm
.idea/

# VS Code
.vscode/
*.code-workspace

# Data - Don't commit large datasets
data/raw/*
data/processed/*
data/cache/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/cache/.gitkeep

# Model weights
*.pt
*.pth
*.onnx
*.engine
*.weights
models/yolov8/*
models/rtdetr/*
!models/yolov8/.gitkeep
!models/rtdetr/.gitkeep

# Results
results/weights/*
results/logs/*
results/predictions/*
!results/weights/.gitkeep
!results/logs/.gitkeep
!results/predictions/.gitkeep

# Temporary files
*.tmp
*.temp
*.log
*.cache
.DS_Store
Thumbs.db

# Kaggle
.kaggle/
kaggle.json

# WandB
wandb/
.wandb/

# Large files
*.zip
*.tar
*.gz
*.rar
*.7z

# Image datasets (keep only samples)
*.png
*.jpg
*.jpeg
*.bmp
*.tiff
!docs/images/*
!examples/*

# Video files
*.mp4
*.avi
*.mov
*.mkv

# Environment variables
.env
.env.local

# Documentation build
docs/_build/
docs/_static/
docs/_templates/

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/

# Profiling
*.prof
*.lprof

# Custom
scratch/
tmp/
temp/
old/
backup/
EOF

# Create .gitkeep files to preserve directory structure
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/cache/.gitkeep
touch models/yolov8/.gitkeep
touch models/rtdetr/.gitkeep
touch results/weights/.gitkeep
touch results/logs/.gitkeep
touch results/predictions/.gitkeep

echo "✓ .gitignore created"