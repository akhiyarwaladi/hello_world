# Troubleshooting Guide - Malaria Detection Project

Complete troubleshooting guide for common issues and their solutions.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Path & Dataset Issues](#path--dataset-issues)
- [GPU & Memory Issues](#gpu--memory-issues)
- [Training Issues](#training-issues)
- [Performance Issues](#performance-issues)

---

## Installation Issues

### 1. Import Errors or Version Mismatches

**Symptoms:**
```
ImportError: cannot import name '...'
ModuleNotFoundError: No module named '...'
```

**Solutions:**
```bash
# Option 1: Reinstall all packages
pip install --upgrade --force-reinstall -r requirements.txt

# Option 2: Run automated setup
python setup_environment.py

# Option 3: Check Python version
python --version  # Should be 3.13.5

# Option 4: Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Prevention:**
- Always use virtual environment (conda or venv)
- Match exact versions in requirements.txt
- Use Python 3.13.5 as specified

---

## Path & Dataset Issues

### 1. Path Errors (data.yaml not found)

**Symptoms:**
```
Error: Dataset 'data/processed/lifecycle/data.yaml' images not found
FileNotFoundError: [Errno 2] No such file or directory
```

**Root Cause:**
- Windows backslash paths (`C:\Users\...`) incompatible with YOLO
- WSL Linux-style paths (`/mnt/c/...`) incompatible on native Windows

**Solution:**
```bash
# Run the automated path fixer
python fix_data_yaml_paths.py

# Manually verify paths are correct
cat data/processed/lifecycle/data.yaml

# Should show: path: C:/Users/.../data/processed/lifecycle (forward slashes)
# NOT: path: C:\Users\...\ (backslashes)
# NOT: path: /mnt/c/Users/... (WSL paths)
```

**Manual Fix:**
```yaml
# Edit data.yaml files directly
path: C:/Users/MyPC PRO/Documents/hello_world/data/processed/lifecycle  # ✅ CORRECT
path: C:\Users\MyPC PRO\Documents\hello_world\data\processed\lifecycle  # ❌ WRONG
path: /mnt/c/Users/MyPC PRO/Documents/hello_world/data/processed/lifecycle  # ❌ WRONG
```

### 2. Missing Datasets

**Symptoms:**
```
Error: Dataset not found
KeyError: 'iml_lifecycle'
```

**Solutions:**
```bash
# 1. Check dataset exists
ls data/processed/lifecycle/
ls data/processed/species/
ls data/processed/stages/

# 2. Re-run dataset setup
python scripts/data_setup/setup_datasets.py

# 3. Manual download (if auto-download fails)
# Download from Kaggle and extract to data/raw/
```

**Requirements:**
- Active internet connection for auto-download
- ~5GB free space for raw datasets
- Kaggle API credentials (for MP-IDB datasets)

---

## GPU & Memory Issues

### 1. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
torch.cuda.OutOfMemoryError
```

**Solutions (Ordered by Impact):**

**Option 1: Reduce Batch Size**
```bash
# Default batch size: 64
# Try: 32, 16, or 8

# Edit main_pipeline.py or classification training script
--batch 32  # or 16, or 8
```

**Option 2: Use Fewer Models**
```bash
# Instead of all models:
python main_pipeline.py --include yolo11 --classification-models densenet121

# Quick test with minimal memory
python main_pipeline.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121 --epochs-det 5 --epochs-cls 5
```

**Option 3: Close Other GPU Applications**
```bash
# Check GPU memory usage
nvidia-smi

# Kill processes using GPU
# Windows: Task Manager → Details → Find python.exe → End task
# Linux: kill -9 <PID>
```

**Option 4: Use CPU (Last Resort)**
```bash
# Slower but works without GPU
export CUDA_VISIBLE_DEVICES=""  # Linux
set CUDA_VISIBLE_DEVICES=  # Windows
```

**Prevention:**
- Monitor GPU memory: `watch -n 1 nvidia-smi` (Linux)
- Close browser/IDE before training
- Use RTX 3060 (12GB) or better
- Batch size 64 works on RTX 4090, use 32 on RTX 3060

### 2. CUDA Not Available

**Symptoms:**
```python
torch.cuda.is_available()  # Returns False
```

**Solutions:**
```bash
# 1. Check CUDA installation
nvidia-smi

# 2. Reinstall PyTorch with correct CUDA version
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 3. Check CUDA compatibility
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# 4. Update NVIDIA drivers
# Download from: https://www.nvidia.com/Download/index.aspx
```

---

## Training Issues

### 1. Long Training Time

**Symptoms:**
- Training takes hours instead of minutes
- Progress very slow (< 1 epoch/minute)

**Solutions:**

**Quick Test (5 minutes):**
```bash
python main_pipeline.py --dataset iml_lifecycle --include yolo11 --classification-models densenet121 --epochs-det 5 --epochs-cls 5
```

**Faster Configuration:**
```bash
# Use fastest YOLO only
python main_pipeline.py --include yolo11

# Use single classification model
python main_pipeline.py --classification-models densenet121

# Reduce epochs
python main_pipeline.py --epochs-det 50 --epochs-cls 30
```

**Performance Tips:**
- YOLO11 is fastest YOLO model
- DenseNet121 is fastest classification model
- Reduce epochs for quick experiments
- Use GPU (10-50x faster than CPU)

### 2. Training Crashes or Hangs

**Symptoms:**
- Process freezes
- No progress for > 5 minutes
- Sudden crash without error

**Solutions:**

**Check Logs:**
```bash
# Look for last output
tail -n 50 training.log
```

**Common Causes & Fixes:**

1. **File Locking (Windows)**
```bash
# Restart training
# New fix (2025-10-17): Automatic folder cleanup with fallback
```

2. **Insufficient Memory**
```bash
# Reduce batch size
# Close other applications
```

3. **Corrupted Data**
```bash
# Re-download datasets
python scripts/data_setup/setup_datasets.py --force
```

### 3. Poor Performance / Low Accuracy

**Symptoms:**
- Detection mAP@50 < 0.70 (70%)
- Classification accuracy < 0.60 (60%)

**Solutions:**

**Check Configuration:**
```bash
# 1. Verify epochs are sufficient
--epochs-det 100  # Default, good
--epochs-cls 75   # Default, good

# 2. Check learning rate
# Default: 0.0005 (should be good)

# 3. Verify data splits
--train-ratio 0.66 --val-ratio 0.17 --test-ratio 0.17
```

**Expected Performance (Baseline):**
- Detection mAP@50: > 0.85 (85%)
- Classification accuracy: > 0.70 (70%)
- Balanced accuracy: > 0.60 (60%)

**If still poor:**
- Check dataset quality (corrupted images?)
- Verify augmentation settings
- Try different models (EfficientNet-B1 usually best)
- Increase epochs (150-200 for detection, 100-150 for classification)

---

## Performance Issues

### 1. Slow Data Loading

**Symptoms:**
- GPU utilization < 50%
- Long wait between batches

**Solutions:**
```python
# Adjust DataLoader workers
num_workers = 4  # Default (Windows optimized)
num_workers = 6  # Linux/WSL

# Enable persistent workers
persistent_workers = True

# Adjust prefetch factor
prefetch_factor = 4  # Default, good
```

### 2. Storage Space Issues

**Symptoms:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**
```bash
# 1. Check disk space
df -h  # Linux
dir   # Windows

# 2. Clean old results
rm -rf results/optA_OLD_TIMESTAMP/

# 3. Clean crops (can regenerate)
rm -rf data/crops_ground_truth/*/

# 4. Use --no-zip to skip archiving
python main_pipeline.py --no-zip
```

**Space Requirements:**
- Raw datasets: ~5GB
- Processed datasets: ~2GB
- Ground truth crops: ~500MB per dataset
- Results per experiment: ~2-5GB
- **Total recommended**: 50GB free

### 3. High CPU Usage

**Symptoms:**
- CPU usage 90-100%
- System slow/unresponsive

**Solutions:**
```bash
# 1. Reduce DataLoader workers
num_workers = 2  # Instead of 4

# 2. Reduce PyTorch threads
torch.set_num_threads(4)  # Instead of 8

# 3. Close other applications

# 4. Use task manager to set priority
# Windows: Task Manager → Details → Right-click python.exe → Set priority → Below normal
```

---

## Advanced Troubleshooting

### Enable Debug Mode

```bash
# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Run with verbose output
python main_pipeline.py --verbose
```

### Check System Configuration

```bash
# Full system check
python -c "
import torch, platform, psutil
print(f'Platform: {platform.system()} {platform.release()}')
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')
print(f'CPU: {psutil.cpu_count()} cores')
"
```

### Get Help

If you encounter issues not covered here:

1. **Check logs**: Look in experiment folder for detailed logs
2. **Check GitHub issues**: https://github.com/akhiyarwaladi/hello_world/issues
3. **Create issue**: Include error message, command used, system info
4. **Contact maintainer**: Include CLAUDE.md version and error logs

---

## Common Error Messages Reference

### Windows Error 1920
**Error:** `PermissionError: [WinError 1920]`
**Solution:** New automatic cleanup with fallback (2025-10-17). If persists, manually delete folder before re-running.

### Ultralytics Path Error
**Error:** `AssertionError: train: No labels found in ...`
**Solution:** Run `python fix_data_yaml_paths.py`

### Checkpoint Load Error
**Error:** `RuntimeError: Error(s) in loading state_dict`
**Solution:** Model architecture changed. Delete checkpoints and retrain.

### PIL Image Error
**Error:** `PIL.Image.DecompressionBombError`
**Solution:** Increase limit: `from PIL import Image; Image.MAX_IMAGE_PIXELS = None`

---

*Last Updated: 2025-10-17*
*For more help: See CLAUDE.md, SETUP_GUIDE.md, or create GitHub issue*
