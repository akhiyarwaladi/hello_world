# 🔧 How to Add New Loss Functions or Models

## 📊 Current Configuration

### **Loss Functions (Already Implemented):**
- ✅ **Cross-Entropy** - Standard classification loss
- ✅ **Focal Loss** - For imbalanced data (α=1.0, γ=2.0)

### **Models (Currently Active):**
- ✅ **DenseNet121**
- ✅ **EfficientNet-B1**
- ✅ **EfficientNet-B2**
- ✅ **ResNet101**
- ✅ **VGG16** (already in pipeline!)

**Total:** 10 models (5 architectures × 2 loss functions)

---

## 🔥 Adding New Loss Functions

### **Option 1: Use Advanced Losses (Already Created!)**

File: `scripts/training/advanced_losses.py`

**Available Losses:**
1. **Label Smoothing CE** - Prevents overconfidence
2. **Class-Balanced Loss** - Automatic class weighting
3. **Weighted Focal Loss** - Enhanced focal loss
4. **Dice Loss** - Medical imaging friendly
5. **Combined Loss** - Mix multiple losses

### **Quick Start - Add Label Smoothing:**

**Step 1:** Modify `run_multiple_models_pipeline_OPTION_A.py`

Find the classification_configs section (around line 820):

```python
base_models = ["densenet121", "efficientnet_b1", "vgg16", "resnet50", "efficientnet_b2", "resnet101"]
```

**Step 2:** Add new loss configuration:

```python
# After focal loss config, add:
# Configuration 3: Label Smoothing (New!)
classification_configs[f"{model}_ls"] = {
    "type": "pytorch",
    "script": "scripts/training/12_train_pytorch_classification.py",
    "model": model,
    "loss": "label_smoothing",
    "smoothing": 0.1,
    "epochs": 25,
    "batch": 32,
    "lr": 0.0005,
    "display_name": f"{model.upper()} (Label Smoothing)"
}
```

**Step 3:** Modify training script to support it

Edit `scripts/training/12_train_pytorch_classification.py`:

```python
# In main() function, after Focal Loss setup:

elif args.loss == 'label_smoothing':
    from scripts.training.advanced_losses import LabelSmoothingCrossEntropy
    smoothing = args.smoothing if hasattr(args, 'smoothing') else 0.1
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    print(f"   Using Label Smoothing Cross-Entropy (smoothing={smoothing})")
```

**Step 4:** Run pipeline:

```bash
python run_multiple_models_pipeline_OPTION_A.py \
  --dataset iml_lifecycle \
  --classification-models densenet121_ls efficientnet_b1_ls
```

---

### **Option 2: Add Class-Balanced Loss**

**Why?** Automatically handles class imbalance based on sample counts.

**Step 1:** Add to classification_configs:

```python
# Configuration 4: Class-Balanced Loss
classification_configs[f"{model}_cb"] = {
    "type": "pytorch",
    "script": "scripts/training/12_train_pytorch_classification.py",
    "model": model,
    "loss": "class_balanced",
    "cb_beta": 0.9999,  # Hyperparameter
    "epochs": 25,
    "batch": 32,
    "lr": 0.0005,
    "display_name": f"{model.upper()} (Class-Balanced)"
}
```

**Step 2:** Modify training script:

```python
elif args.loss == 'class_balanced':
    from scripts.training.advanced_losses import ClassBalancedLoss

    # Calculate samples per class
    from collections import Counter
    all_labels = [label for _, label in train_dataset]
    class_counts = Counter(all_labels)
    samples_per_class = [class_counts[i] for i in range(num_classes)]

    cb_beta = args.cb_beta if hasattr(args, 'cb_beta') else 0.9999
    criterion = ClassBalancedLoss(samples_per_class, beta=cb_beta)
    print(f"   Using Class-Balanced Loss (beta={cb_beta})")
    print(f"   Samples per class: {samples_per_class}")
```

---

## 🤖 Adding New Models

### **Option 1: Activate Existing Models (Easy!)**

**Available but not active:**
- ResNet18, ResNet34, ResNet50
- DenseNet161, DenseNet169
- MobileNet-V2, MobileNet-V3-Small, MobileNet-V3-Large
- VGG19
- ViT-B/16, ViT-B/32

**To activate, edit `base_models` in `run_multiple_models_pipeline_OPTION_A.py`:**

```python
# Current:
base_models = ["densenet121", "efficientnet_b1", "vgg16", "resnet50", "efficientnet_b2", "resnet101"]

# Add more:
base_models = [
    "densenet121",
    "efficientnet_b1",
    "efficientnet_b2",
    "vgg16",
    "resnet50",
    "resnet101",
    "mobilenet_v3_large",  # NEW
    "vit_b_16",            # NEW - Vision Transformer!
]
```

**That's it!** Pipeline will automatically train all combinations.

---

### **Option 2: Add ConvNeXt (Modern CNN)**

ConvNeXt is mentioned in docs but not implemented. Let's add it:

**Step 1:** Add to `get_model()` in `12_train_pytorch_classification.py`:

```python
# Add after densenet section (around line 126):

elif model_name.startswith('convnext'):
    if model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)
    elif model_name == 'convnext_small':
        model = models.convnext_small(weights='IMAGENET1K_V1' if pretrained else None)
    elif model_name == 'convnext_base':
        model = models.convnext_base(weights='IMAGENET1K_V1' if pretrained else None)
    else:
        raise ValueError(f"Unknown ConvNeXt model: {model_name}")

    # Modify final layer
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
```

**Step 2:** Add to base_models:

```python
base_models = [
    "densenet121",
    "efficientnet_b1",
    "efficientnet_b2",
    "vgg16",
    "resnet101",
    "convnext_tiny",  # NEW!
]
```

---

### **Option 3: Add Swin Transformer (State-of-the-art!)**

**Step 1:** Install if needed:

```bash
pip install timm
```

**Step 2:** Add to `get_model()`:

```python
elif model_name.startswith('swin'):
    if model_name == 'swin_t':
        model = models.swin_t(weights='IMAGENET1K_V1' if pretrained else None)
    elif model_name == 'swin_s':
        model = models.swin_s(weights='IMAGENET1K_V1' if pretrained else None)
    elif model_name == 'swin_b':
        model = models.swin_b(weights='IMAGENET1K_V1' if pretrained else None)
    else:
        raise ValueError(f"Unknown Swin model: {model_name}")

    # Modify final layer
    model.head = nn.Linear(model.head.in_features, num_classes)
```

**Step 3:** Add to base_models and run!

---

## 📋 Testing New Additions

### **Test Single Model + Loss:**

```bash
# Test Label Smoothing with DenseNet121
python scripts/training/12_train_pytorch_classification.py \
  --model densenet121 \
  --loss label_smoothing \
  --smoothing 0.1 \
  --epochs 5 \
  --data data/ground_truth_crops_224/lifecycle \
  --output test_label_smoothing
```

### **Test in Pipeline:**

```bash
# Quick test with new configuration
python run_multiple_models_pipeline_OPTION_A.py \
  --dataset iml_lifecycle \
  --include yolo11 \
  --classification-models densenet121_ls \
  --epochs-det 5 \
  --epochs-cls 5 \
  --no-zip
```

---

## 🎯 Recommended Combinations

### **For Better Performance:**

1. **Label Smoothing + DenseNet121**
   - Prevents overconfidence
   - Good generalization

2. **Class-Balanced + EfficientNet-B1**
   - Handles imbalance automatically
   - Efficient architecture

3. **Weighted Focal + Swin-T**
   - State-of-the-art transformer
   - Strong feature extraction

### **For Speed:**

1. **Cross-Entropy + MobileNet-V3-Large**
   - Fast inference
   - Good accuracy

2. **Focal Loss + EfficientNet-B0**
   - Smaller than B1
   - Still robust to imbalance

---

## ⚠️ Important Notes

### **When Adding New Loss:**
1. ✅ Test standalone first
2. ✅ Check for NaN losses
3. ✅ Verify gradients flow correctly
4. ✅ Compare with baseline (CE)

### **When Adding New Model:**
1. ✅ Ensure input size is 224×224
2. ✅ Modify correct final layer
3. ✅ Test with small epochs first
4. ✅ Check memory usage (GPU)

### **Best Practices:**
- Start with 1-2 new configurations
- Test on single dataset first
- Monitor training curves
- Compare with existing baselines

---

## 📊 Example: Full New Configuration

Here's a complete example adding **Label Smoothing + Swin-T**:

### **1. Modify `12_train_pytorch_classification.py`:**

```python
# Add Swin model support
elif model_name.startswith('swin'):
    model = models.swin_t(weights='IMAGENET1K_V1' if pretrained else None)
    model.head = nn.Linear(model.head.in_features, num_classes)

# Add Label Smoothing support
elif args.loss == 'label_smoothing':
    from scripts.training.advanced_losses import LabelSmoothingCrossEntropy
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

### **2. Modify `run_multiple_models_pipeline_OPTION_A.py`:**

```python
# Add to base_models
base_models = [..., "swin_t"]

# Add to classification_configs
classification_configs["swin_t_ls"] = {
    "type": "pytorch",
    "model": "swin_t",
    "loss": "label_smoothing",
    "smoothing": 0.1,
    ...
}
```

### **3. Run:**

```bash
python run_multiple_models_pipeline_OPTION_A.py \
  --dataset iml_lifecycle \
  --classification-models swin_t_ls
```

---

## ✅ Summary

**Currently Have:**
- 2 loss functions (CE, Focal)
- 5 models (DenseNet121, EfficientNet-B1/B2, ResNet101, VGG16)

**Can Easily Add:**
- 5+ loss functions (Label Smoothing, Class-Balanced, etc.)
- 10+ models (ViT, Swin, ConvNeXt, MobileNet, etc.)

**Files to Modify:**
1. `scripts/training/12_train_pytorch_classification.py` - Add model/loss support
2. `run_multiple_models_pipeline_OPTION_A.py` - Add to configurations
3. `scripts/training/advanced_losses.py` - Already has 5 new losses!

**Next Steps:**
1. Choose loss function OR model to add
2. Follow steps above
3. Test with small dataset first
4. Run full pipeline if successful

---

**Created:** 2025-10-02
**Script:** `scripts/training/advanced_losses.py` (NEW!)
**Models Available:** 15+ architectures ready to use
**Losses Available:** 7+ loss functions ready to use
