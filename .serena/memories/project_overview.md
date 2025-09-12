# Malaria Detection Project Overview

## Project Purpose
Multi-species malaria parasite detection and classification using YOLOv8 and RT-DETR deep learning models. The project processes various malaria datasets from different sources to create a unified dataset for training detection models.

## Tech Stack
- **Deep Learning Framework**: PyTorch, Ultralytics YOLOv8, RT-DETR
- **Computer Vision**: OpenCV, PIL, matplotlib
- **Data Processing**: NumPy, Pandas, scikit-learn, Albumentations
- **Utilities**: PyYAML, tqdm, requests, BeautifulSoup, Kaggle API
- **Visualization**: Matplotlib, Seaborn
- **Experiment Tracking**: Weights & Biases (wandb)

## Dataset Sources
1. **NIH Cell Images**: 27,558 segmented cells (2 classes)
2. **MP-IDB**: 210 whole slide images (4 species)
3. **BBBC041**: 1,364 P. vivax stage images
4. **PlasmoID**: 559 Indonesian dataset images (4 species)
5. **IML**: 345 P. vivax life cycle images
6. **Uganda**: 4,000 mixed images

## Target Classes
- P_falciparum (Class 0)
- P_vivax (Class 1) 
- P_malariae (Class 2)
- P_ovale (Class 3)
- Mixed_infection (Class 4)
- Uninfected (Class 5)

## Project Structure
- `/scripts/`: Data processing pipeline (01-06 numbered scripts)
- `/config/`: YAML configuration files
- `/data/`: Raw and processed datasets
- `/models/`: Model weights and checkpoints
- `/results/`: Training results, logs, predictions
- `/notebooks/`: Jupyter notebooks for analysis
- `/tests/`: Unit tests