# 🔬 Malaria Detection Pipeline
> Two-Step Classification Approach: YOLOv8/YOLOv11/RT-DETR Detection + CNN Classification

## 📊 Project Status (Updated: September 13, 2025)

### ✅ **Pipeline Complete - Ready for Research**
- **Data Pipeline**: ✅ FULLY IMPLEMENTED - 6 datasets integrated
- **Detection Dataset**: ✅ 103 images with 1,242 parasites (MP-IDB with fixed coordinates)
- **Classification Dataset**: ✅ 1,242 cropped parasites (128x128px) for single-cell classification
- **Two-Step Approach**: ✅ Detection → Cropping → Classification pipeline verified
- **Model Training**: ⚠️ Scripts ready, partial training completed (interrupted)
- **Research Paper**: 📄 "Perbandingan YOLOv8, YOLOv11, dan RT-DETR untuk Deteksi Malaria"

### 🎯 **Key Contributions**
- **Fixed MP-IDB Bounding Boxes**: Corrected coordinate mapping using ground truth masks
- **Two-Step Classification**: YOLO detection + CNN classification for species identification
- **Complete Pipeline**: End-to-end automation from download to training
- **Multi-Model Comparison**: YOLOv8, YOLOv11, RT-DETR performance analysis

### 📈 **Research Results**
- **Detection Approach**: Parasite localization in microscopy images
- **Classification Approach**: Species identification on cropped single cells
- **Dataset Quality**: High-quality 1,242 parasite annotations with corrected coordinates
- **Reproducible Pipeline**: Complete automation for research replication

## 🚀 Unified Command-Line Interface

`pipeline.py` kini membaca konfigurasi dari `config/models.yaml` sehingga Anda cukup memilih model yang sudah terdaftar.

```bash
# Lihat model yang siap dijalankan
python pipeline.py list --detailed

# Latih model sesuai konfigurasi (override parameter bila perlu)
python pipeline.py train yolov8_detection --name demo_det
python pipeline.py train yolov10_detection --name demo_det_v10 --set model=yolov10s.pt
python pipeline.py train yolov8_classification --name demo_cls --set epochs=10
python pipeline.py train pytorch_resnet18_classification --name demo_resnet

# Gunakan --dry-run untuk melihat perintah tanpa mengeksekusi
python pipeline.py train yolov11_detection --name debug --dry-run
```

Argumen umum yang bisa dioverride langsung: `--data`, `--epochs`, `--batch`, `--device`, `--imgsz`, `--model-weights`, atau gunakan `--set key=value` untuk opsi tambahan.

**Deteksi yang tersedia:** `yolov8_detection`, `yolov10_detection`, `yolov11_detection`, `rtdetr_detection`  
**Klasifikasi yang tersedia:** `yolov8_classification`, `yolov11_classification`, `pytorch_resnet18_classification`, `pytorch_efficientnet_b0_classification`, `pytorch_densenet121_classification`, `pytorch_mobilenet_v2_classification`

## 📁 Repository Structure

```
malaria-yolo-detection/
│
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore file
├── LICENSE                      # MIT License
│
├── config/
│   ├── dataset_config.yaml     # Dataset configuration
│   ├── yolo_config.yaml        # YOLO training configuration
│   └── class_names.yaml        # Class definitions
│
├── data/
│   ├── raw/                    # Raw downloaded datasets
│   │   ├── nih_cell/
│   │   ├── mp_idb/
│   │   ├── bbbc041/
│   │   ├── plasmoID/
│   │   ├── iml/
│   │   └── uganda/
│   │
│   ├── processed/              # Processed and integrated data
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   │
│   └── cache/                  # Temporary files
│
├── scripts/
│   ├── 01_download_datasets.py     # Download all datasets ✅
│   ├── 02_preprocess_data.py       # Preprocess and standardize 🔄
│   ├── 03_integrate_datasets.py    # Integrate all datasets ✅🔄
│   ├── 04_convert_to_yolo.py       # Convert to YOLO format
│   ├── 05_augment_data.py          # Data augmentation
│   ├── 06_split_dataset.py         # Train/val/test split
│   ├── 07_train_yolo_detection.py  # YOLOv8 detection training
│   ├── 08_train_yolo11_detection.py# YOLOv11 detection training
│   ├── 09_train_rtdetr_detection.py# RT-DETR detection training
│   ├── 11_train_classification_crops.py  # YOLO classification training
│   ├── 11b_train_pytorch_classification.py # Torch-based classifiers
│   ├── 12_generate_crops_from_detection.py # Auto crop generation
│   ├── 13_full_detection_classification_pipeline.py # Detection→classification orchestrator
│   ├── run_full_pipeline.py        # Manual data pipeline execution
│   └── watch_pipeline.py           # Automated pipeline monitoring 🤖
│   └── utils/
│       ├── __init__.py
│       ├── download_utils.py       # Download helper functions
│       ├── image_utils.py          # Image processing utilities
│       └── annotation_utils.py     # Annotation conversion utilities
│
├── models/
│   ├── yolov8/                     # YOLOv8 models
│   └── rtdetr/                     # RT-DETR models
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Data analysis
│   ├── 02_visualization.ipynb     # Visualization tools
│   └── 03_model_evaluation.ipynb  # Model evaluation
│
├── results/
│   ├── weights/                    # Trained model weights
│   ├── logs/                       # Training logs
│   └── predictions/                # Prediction results
│
└── tests/
    ├── test_download.py            # Unit tests
    └── test_preprocessing.py       # Preprocessing tests
```

## 📋 Requirements

```txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
albumentations>=1.3.0
tqdm>=4.65.0
pyyaml>=6.0
requests>=2.31.0
gdown>=4.7.0
kaggle>=1.5.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
scipy>=1.11.0
labelme>=5.3.0
shapely>=2.0.0
```

## 🎯 Class Configuration

```yaml
# config/class_names.yaml
classes:
  0: "P_falciparum"      # Plasmodium falciparum infected
  1: "P_vivax"           # Plasmodium vivax infected
  2: "P_malariae"        # Plasmodium malariae infected
  3: "P_ovale"           # Plasmodium ovale infected
  4: "Mixed_infection"   # Multiple species infection
  5: "Uninfected"        # Healthy/uninfected cells

colors:  # BGR format for visualization
  0: [255, 0, 0]       # Blue
  1: [0, 255, 0]       # Green
  2: [0, 0, 255]       # Red
  3: [255, 255, 0]     # Cyan
  4: [255, 0, 255]     # Magenta
  5: [128, 128, 128]   # Gray
```

## 🔧 Installation & Setup

### **🚀 Quick Setup (Recommended for New Machines)**
```bash
# 1. Clone repository
git clone https://github.com/akhiyarwaladi/malaria_detection.git
cd malaria_detection

# 2. Run automated setup script (handles everything)
chmod +x quick_setup_new_machine.sh
./quick_setup_new_machine.sh
```

**What the script does:**
- ✅ Creates virtual environment and installs dependencies
- ✅ Downloads all 6 datasets (~30 mins)
- ✅ Prepares detection dataset (103 images, 1,242 parasites)
- ✅ Crops individual parasites (1,242 single cells)
- ✅ Runs training tests to verify system
- ✅ Generates verification documentation

### **🔍 Manual Setup (Step by Step)**
```bash
# 1. Clone repository
git clone https://github.com/akhiyarwaladi/malaria_detection.git
cd malaria_detection

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch, ultralytics, cv2; print('✅ Setup successful')"
```

### **📋 Setup Verification**
```bash
# Use detailed verification checklist
# See: setup_verification.md (comprehensive 5-phase verification)

# Quick verification commands:
echo "Datasets: $(ls data/raw/ | wc -l)"              # Should be 6+
echo "Detection images: $(ls data/detection_fixed/images/ | wc -l)"  # Should be 103
echo "Parasites: $(find data/classification_crops/ -name "*.jpg" | wc -l)"  # Should be 1,242
```

## 📊 Dataset Information

| Dataset | Images | Classes | Resolution | Size |
|---------|--------|---------|------------|------|
| NIH Cell Images | 27,558 | 2 | Variable | ~350MB |
| MP-IDB | 210 | 4 species | 2592×1944 | ~1.2GB |
| BBBC041 | 1,364 | P. vivax stages | 1000x | ~800MB |
| PlasmoID | 559 | 4 species | 960×1280 | ~500MB |
| IML | 345 | P. vivax stages | 100x | ~300MB |
| Uganda | 4,000 | Mixed | Variable | ~2GB |

## 🚀 Complete Pipeline Flow

### **Phase 1: Data Download**

#### **Option A: MP-IDB Only (RECOMMENDED)** ⏱️ ~5-10 mins
```bash
# Download only MP-IDB (sufficient for main research pipeline)
python scripts/01_download_datasets.py --dataset mp_idb

# Expected output: data/raw/mp_idb/
# Size: ~500MB, Contains: 103 images with 1,242 parasites
```

#### **Option B: All Datasets (Comprehensive)** ⏱️ ~30-60 mins
```bash
# Download all 6 datasets (for full research)
python scripts/01_download_datasets.py --dataset all

# Expected output: data/raw/ with 6 datasets (~6GB)
```

#### **Option C: Custom Selection**
```bash
# Multiple specific datasets
python scripts/01_download_datasets.py --dataset mp_idb,nih_cell

# List available datasets
python scripts/01_download_datasets.py --list-datasets
```

### **Phase 2: Detection & Crop Dataset Preparation** ⏱️ ~10 mins
```bash
# Bangun ulang dataset deteksi + klasifikasi multispecies lengkap
python create_multispecies_dataset.py

# atau gunakan deteksi terlatih untuk membuat crop baru
python scripts/12_generate_crops_from_detection.py \
    --model results/.../weights/best.pt \
    --input data/detection_multispecies \
    --output data/crops_from_yolo8_detection --create_yolo_structure

# Output utama:
# - data/detection_multispecies/ (YOLO detection format)
# - data/classification_multispecies/ (species-aware crops)
# - data/crops_from_*/yolo_classification/ (pipa deteksi→klasifikasi)
```

### **Phase 4: Model Training** ⏱️ ~2-8 hours
```bash
# Detection models
python pipeline.py train yolov8_detection --name yolov8_det --set epochs=30
python pipeline.py train yolov10_detection --name yolov10_det --set epochs=30 --set model=yolov10s.pt
python pipeline.py train yolov11_detection --name yolov11_det --set epochs=20
python pipeline.py train rtdetr_detection --name rtdetr_det --set epochs=20

# Classification models
python pipeline.py train yolov8_classification --name yolo8_cls --set epochs=25
python pipeline.py train yolov11_classification --name yolo11_cls --set epochs=25
python pipeline.py train pytorch_resnet18_classification --name resnet18_cls --set epochs=25
python pipeline.py train pytorch_efficientnet_b0_classification --name effnet_cls --set epochs=25
python pipeline.py train pytorch_densenet121_classification --name densenet_cls --set epochs=25
python pipeline.py train pytorch_mobilenet_v2_classification --name mobilenet_cls --set epochs=25
```

### **Phase 5: Performance Analysis** ⏱️ ~2 mins
```bash
# Generate comprehensive comparison report
python scripts/14_compare_models_performance.py

# Output: results/model_comparison_report.md
```

## 📈 Training Configuration

```yaml
# config/yolo_config.yaml
path: ../data/processed
train: images/train
val: images/val
test: images/test

nc: 6  # number of classes
names: ['P_falciparum', 'P_vivax', 'P_malariae', 'P_ovale', 'Mixed_infection', 'Uninfected']

# Training hyperparameters
batch: 16
imgsz: 640
epochs: 100
patience: 50
optimizer: 'AdamW'
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
close_mosaic: 10
mixup: 0.0
copy_paste: 0.0
```

## 📝 Dataset Configuration

```yaml
# config/dataset_config.yaml
datasets:
  nih_cell:
    url: "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
    type: "segmented_cells"
    classes: ["infected", "uninfected"]
    format: "folder"
    
  nih_thick_pf:
    url: "https://data.lhncbc.nlm.nih.gov/public/Malaria/Thick_Smears_150/"
    type: "whole_slide"
    species: "P_falciparum"
    format: "custom"
    
  nih_thick_pv:
    url: "https://data.lhncbc.nlm.nih.gov/public/Malaria/NIH-NLM-ThickBloodSmearsPV/NIH-NLM-ThickBloodSmearsPV.zip"
    type: "whole_slide"
    species: "P_vivax"
    format: "custom"
    
  mp_idb:
    url: "https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis.git"
    type: "whole_slide"
    classes: ["P_falciparum", "P_vivax", "P_malariae", "P_ovale"]
    format: "custom_annotation"
    
  bbbc041:
    url: "https://data.broadinstitute.org/bbbc/BBBC041/"
    type: "whole_slide"
    species: "P_vivax"
    stages: ["ring", "trophozoite", "schizont", "gametocyte"]
    format: "matlab"
    
  kaggle_nih:
    dataset: "iarunava/cell-images-for-detecting-malaria"
    type: "segmented_cells"
    classes: ["parasitized", "uninfected"]
    format: "folder"

augmentation:
  minority_threshold: 500  # Minimum samples per class
  techniques:
    - rotation: [-30, 30]
    - flip: ["horizontal", "vertical"]
    - brightness: [0.8, 1.2]
    - contrast: [0.8, 1.2]
    - blur: [0, 2]
    - noise: 0.02
```

## 👥 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NIH/NLM for the comprehensive malaria datasets
- MP-IDB authors (Loddo et al.)
- Broad Institute for BBBC041
- PlasmoID team for Indonesian dataset
- All researchers who made their data publicly available

## 📧 Contact

- Your Name - [your.email@example.com]
- Project Link: [https://github.com/yourusername/malaria-yolo-detection]

## 📊 Two-Step Classification Results

### **Research Paper Framework**
```bibtex
Title: "Perbandingan YOLOv8, YOLOv11, dan RT-DETR untuk Deteksi Parasit Malaria"
Approach: Two-step classification (Detection → Single-cell Classification)
Dataset: 1,242 P. falciparum parasites from corrected MP-IDB annotations
```

### **Pipeline Verification**
- ✅ **Detection Dataset**: 103 images, 1,242 parasites
- ✅ **Cropped Dataset**: 1,242 single-cell parasites (128x128px)
- ✅ **Training Scripts**: YOLOv8, YOLOv11, RT-DETR detection + Classification
- ✅ **Performance Analysis**: Automated comparison report generation

### **Expected Performance Metrics**
| Approach | Task | Dataset | Expected mAP/Accuracy |
|----------|------|---------|---------------------|
| YOLOv8 | Detection | 103 images | mAP50: 0.85+ |
| YOLOv11 | Detection | 103 images | mAP50: 0.87+ |
| RT-DETR | Detection | 103 images | mAP50: 0.89+ |
| YOLOv8-cls | Classification | 1,242 crops | Accuracy: 0.90+ |

## 🔍 Citation

If you use this dataset or code in your research, please cite:

```bibtex
@misc{malaria-yolo-2024,
  title={Multi-Species Malaria Parasite Detection using Deep Learning},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/malaria-yolo-detection}
}
```
