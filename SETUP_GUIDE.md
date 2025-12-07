# Environment Setup Guide - Malaria Detection Pipeline

## Quick Start - NEW ENVIRONMENT (Recommended untuk Transfer ke PC Lain)

### Option 1: Automated Setup dengan Script Baru (RECOMMENDED)

Script ini akan membuat environment conda BARU dengan semua dependencies:

```bash
# Buka Anaconda Prompt
cd "C:\Users\MyPC PRO\Documents\hello_world"
python setup_new_environment.py
```

Atau dengan nama environment custom:
```bash
python setup_new_environment.py --name my_malaria_env
```

Script ini akan:
1. Membuat conda environment baru: `malaria_detection`
2. Install PyTorch 2.8.0 dengan CUDA 12.8
3. Install Ultralytics (YOLO 10, 11, 12)
4. Install semua dependencies dari requirements.txt
5. Verifikasi instalasi
6. Membuat script aktivasi (activate_env.bat)

**Estimated time:** 10-15 menit
**Disk space:** ~10GB

---

### Option 2: Manual Setup dengan Conda Environment Baru

Langkah-langkah manual jika script otomatis gagal:

#### Step 1: Buat Conda Environment Baru

```bash
# Buka Anaconda Prompt
conda create -n malaria_detection python=3.11 -y
conda activate malaria_detection
```

#### Step 2: Install PyTorch dengan CUDA 12.8

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Verifikasi CUDA:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```
Expected output: `CUDA Available: True`

#### Step 3: Install Ultralytics (YOLO)

```bash
pip install ultralytics==8.3.202
```

#### Step 4: Install Dependencies

```bash
cd "C:\path\to\hello_world"
pip install -r requirements.txt
```

Atau gunakan minimal requirements (lebih cepat):
```bash
pip install -r requirements_minimal.txt
```

---

## Transfer ke PC Lain

### Cara Transfer Project:

1. **Copy folder project** ke PC baru (zip dulu jika perlu)
2. **Install Anaconda** di PC baru jika belum ada
3. **Jalankan setup script:**
   ```bash
   cd "path/to/hello_world"
   python setup_new_environment.py
   ```
4. **Aktifkan environment:**
   ```bash
   conda activate malaria_detection
   ```
5. **Test pipeline:**
   ```bash
   python main_pipeline.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121 --epochs-det 5 --epochs-cls 5
   ```

### File yang HARUS di-copy:
- Semua file Python (*.py)
- requirements.txt dan requirements_minimal.txt
- environment.yml
- setup_new_environment.py
- Folder scripts/, utils/, config/
- Folder data/ (jika sudah ada dataset)

---

## Option 3: Menggunakan Base Environment (NOT RECOMMENDED)

Jika ingin install di base environment (tidak disarankan):

```bash
conda activate base
cd "C:\Users\MyPC PRO\Documents\hello_world"
python setup_environment.py
```

**WARNING:** Ini bisa menyebabkan konflik package. Lebih baik gunakan environment terpisah.

---

## Dependencies yang Diinstall

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.8.0 | Deep learning framework |
| torchvision | 0.23.0 | Vision utilities |
| ultralytics | 8.3.202 | YOLO 10, 11, 12 models |
| opencv-python | 4.12.0.88 | Image processing |
| pandas | 2.3.2 | Data analysis |
| numpy | 2.2.6 | Numerical computing |
| scikit-learn | 1.7.2 | Machine learning metrics |
| albumentations | 2.0.8 | Data augmentation |
| matplotlib | 3.10.6 | Visualization |
| seaborn | 0.13.2 | Statistical plots |
| openpyxl | 3.1.0+ | Excel file writing |
| xlsxwriter | 3.1.0+ | Excel file writing (backup) |

Full list: Lihat `requirements.txt` (~100 packages)

---

## Verification

After installation, verify all packages:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

Expected output:
```
PyTorch: 2.8.0, CUDA: True
Ultralytics: 8.3.202
OpenCV: 4.12.0.88
Pandas: 2.3.2
```

---

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 3060 or better (12GB+ VRAM recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ free space for datasets and results

### Software
- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.10+ (via Anaconda/Miniconda)
- **CUDA**: 12.8 (installed with PyTorch)
- **Conda**: 25.5.1+

---

## Running the Pipeline

### Quick Test (5 minutes)

```bash
python main_pipeline.py \
  --dataset iml_lifecycle \
  --include yolo11 \
  --classification-models densenet121 \
  --epochs-det 5 \
  --epochs-cls 5 \
  --no-zip
```

This runs:
- 1 dataset (IML Lifecycle)
- 1 detection model (YOLO11)
- 1 classification model (DenseNet121)
- 5 epochs each (quick test)

### Single Dataset Full Experiment (2-3 hours)

```bash
python main_pipeline.py \
  --dataset iml_lifecycle \
  --epochs-det 100 \
  --epochs-cls 75
```

This runs:
- 1 dataset (IML Lifecycle)
- 3 detection models (YOLO10, YOLO11, YOLO12)
- 6 classification models (6 architectures with Focal Loss)
- Full training epochs

### Multi-Dataset Full Experiment (6-8 hours)

```bash
python main_pipeline.py
```

This runs:
- 4 datasets (IML Lifecycle, MP-IDB Species, MP-IDB Stages, MD_2019 Stages)
- 3 detection models per dataset
- 6 classification models per dataset
- Full training epochs

---

## Troubleshooting

### Issue 1: CUDA not available

**Symptoms**: `torch.cuda.is_available()` returns `False`

**Solution**:
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
   ```

### Issue 2: Out of memory errors

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size in `main_pipeline.py` (line 86, 64, 24, 20)
- Use fewer models: `--include yolo11 --classification-models densenet121`
- Close other GPU applications

### Issue 3: Package conflicts

**Symptoms**: Import errors or version mismatches

**Solution**:
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

### Issue 4: Missing datasets

**Symptoms**: `[ERROR] Dataset not found`

**Solution**: Pipeline auto-downloads datasets. Ensure:
- Internet connection is active
- ~5GB free space for raw datasets
- Kaggle API configured (if using MP-IDB datasets)

---

## Package List (Key Dependencies)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.8.0 | Deep learning framework |
| torchvision | 0.23.0 | Vision utilities |
| ultralytics | 8.3.202 | YOLO 10, 11, 12 models |
| opencv-python | 4.12.0.88 | Image processing |
| pandas | 2.3.2 | Data analysis |
| numpy | 2.2.6 | Numerical computing |
| scikit-learn | 1.7.2 | Machine learning |
| albumentations | 2.0.8 | Data augmentation |
| matplotlib | 3.10.6 | Visualization |
| seaborn | 0.13.2 | Statistical plots |

Full list: See `requirements.txt` (96 packages)

---

## Next Steps

After successful setup:

1. **Test environment**: Run quick test pipeline
2. **Review documentation**: Read `CLAUDE.md` for full pipeline details
3. **Run experiments**: Start with single dataset, then expand to multi-dataset
4. **Monitor results**: Check `results/` folder for outputs

---

## Support

For issues or questions:
- Check `CLAUDE.md` for pipeline documentation
- Review error logs in terminal output
- Check GPU memory: `nvidia-smi`
- Verify disk space: `dir` or `df -h`

---

**Last Updated**: 2025-10-16
**Python**: 3.10+
**Conda**: 25.5.1+
**CUDA**: 12.8
