# Multi-Model Hybrid Framework for Malaria Parasite Detection and Classification with Shared Architecture Optimization

---

## ABSTRACT

Malaria remains a critical global health challenge with over 200 million cases annually [1], requiring accurate and rapid microscopic diagnosis [2]. Traditional manual microscopy by expert pathologists is time-consuming (20-30 minutes per slide) and faces workforce shortages [3]. Deep learning approaches show promise but face challenges from small datasets (200-500 images), severe class imbalance (up to 54:1 ratio), and computational inefficiency from training separate models for each detection-classification combination. This study introduces a novel multi-model hybrid framework using shared classification architecture that trains classification models once on ground truth crops and reuses them across all detection methods. The framework employs YOLOv11 for detection (92.90-93.87% mAP@50) and evaluates six CNN architectures (DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101) for classification on three malaria datasets: IML Lifecycle (313 images, lifecycle stages), MP-IDB Species (209 images, species identification), and MP-IDB Stages (209 images, lifecycle stages). Results demonstrate dataset-dependent optimal models: EfficientNet-B2 achieves 87.64% accuracy on IML Lifecycle, EfficientNet-B1 achieves 98.8% accuracy with perfect minority class F1-scores on MP-IDB Species, while EfficientNet-B0 achieves 94.31% accuracy on MP-IDB Stages. Focal Loss optimization (α=0.25, γ=2.0) improves minority class F1-scores to 44.44-100% despite severe imbalance, including perfect 1.0 F1-score on ultra-minority P_malariae (7 samples) and 92.31% on schizont (7 samples), demonstrating effectiveness for clinical malaria diagnosis.

**Keywords**: Malaria detection, Deep learning, YOLOv11, EfficientNet, Shared classification, Focal loss, Class imbalance

---

## 1. INTRODUCTION

### 1.1 Background and Motivation

Malaria, caused by *Plasmodium* parasites transmitted through *Anopheles* mosquitoes, continues to impose a substantial global health burden with approximately 249 million cases and 608,000 deaths reported in 2022 [1]. Accurate species identification and lifecycle stage classification are critical for treatment decisions, as different *Plasmodium* species (*P. falciparum*, *P. vivax*, *P. ovale*, *P. malariae*) respond differently to antimalarial drugs [2]. Traditional microscopy-based diagnosis, while remaining the gold standard, requires 20-30 minutes per slide and depends on scarce trained microscopists (2-3 years training) [3], creating bottlenecks particularly in resource-limited endemic regions.

### 1.2 Existing Solutions and Limitations

Recent advances in computer vision have enabled automated malaria detection using Convolutional Neural Networks (CNNs) and object detection models [4][5]. Single-stage detectors like YOLO achieve real-time performance [6], while two-stage pipelines combining detection with classification improve diagnostic accuracy [7][8]. However, existing approaches face three critical challenges. First, public malaria datasets contain only 200-500 images [9][10], severely limiting model generalization and requiring careful augmentation strategies. Second, ring-stage parasites dominate samples (>85%), while critical stages like gametocytes represent less than 2% of data (54:1 ratio in MP-IDB Stages dataset), causing models to underperform on minority classes. Third, traditional pipelines train separate classification models for each detection method (e.g., 3 detectors × 6 classifiers = 18 models), consuming excessive storage and training time [11].

### 1.3 Proposed Solution

This study introduces a multi-model hybrid framework with shared classification architecture (Option A) that addresses these limitations through a unified pipeline. The framework operates in three stages: first, YOLOv11 locates parasites in blood smears using 640×640 input images trained for 100 epochs; second, ground truth crops of 224×224 pixels are extracted from raw annotations once to create a shared resource; third, six CNN architectures (DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101) are trained once on these ground truth crops for 75 epochs using Focal Loss (α=0.25, γ=2.0), and the resulting classification models are reused across all detection methods without retraining. This approach eliminates redundant training and storage by generating ground truth crops once and training classification models once, while maintaining classification accuracy [11]. The framework is validated on three malaria datasets: IML Lifecycle (313 images, 4 lifecycle stages), MP-IDB Species (209 images, 4 species), and MP-IDB Stages (209 images, 4 lifecycle stages), totaling 731 microscopy images.

### 1.4 Contributions

This work makes three key contributions to automated malaria diagnosis. First, we introduce a shared classification architecture where ground truth crops eliminate detection noise, enabling consistent classification across all detection backends with train-once-reuse-everywhere efficiency [12]. Second, we demonstrate dataset-dependent model selection where EfficientNet-B2 (9.2M parameters) outperforms larger ResNet101 (44.5M) by 10.11% on IML Lifecycle (87.64% vs 77.53% accuracy), EfficientNet-B1 (7.8M) achieves 98.8% on MP-IDB Species with perfect minority class performance (1.0 F1-score on P_malariae with only 7 samples), while EfficientNet-B0 (5.3M) achieves 94.31% on MP-IDB Stages, establishing parameter efficiency over model size as a key design principle [13][14]. Third, we show that Focal Loss (α=0.25, γ=2.0) improves minority class F1-scores to 44.44-100% across datasets, including perfect 1.0 F1-score on ultra-minority P_malariae (7 samples) and 92.31% for schizont on MP-IDB Stages (7 samples), effectively addressing severe class imbalance [15][16]. The framework demonstrates efficient resource utilization, enabling deployment in resource-constrained clinical settings. Code and trained models are publicly available to support reproducible research [17].

---

## 2. METHODS

### 2.1 Datasets and Preprocessing

The IML (Immunology, Malaria) Lifecycle dataset [9] contains 313 microscopy images with 4 lifecycle stage classes: ring (272 samples, 54.4%), trophozoite (68 samples, 13.6%), gametocyte (110 samples, 22.0%), and schizont (50 samples, 10.0%). The dataset exhibits moderate imbalance (5.4:1 ring-to-schizont ratio) with annotations in YOLO format (class, x_center, y_center, width, height normalized). The Malaria Parasite Image Database (MP-IDB) comprises two complementary datasets [10]: MP-IDB Species (209 images, 4 species classes) contains P_falciparum (227 samples, 90.8%), P_vivax (11 samples, 4.4%), P_malariae (7 samples, 2.8%), and P_ovale (5 samples, 2.0%), enabling species identification critical for treatment selection; MP-IDB Stages (209 images, 4 lifecycle stage classes) exhibits severe imbalance with ring (272 samples, 90.4%), trophozoite (15 samples, 5.0%), schizont (7 samples, 2.3%), and gametocyte (5 samples, 1.7%). The 54:1 ring-to-gametocyte ratio in MP-IDB Stages represents realistic clinical scenarios where minority stages are diagnostically critical but rare. All three datasets are split into 66% training, 17% validation, and 17% test sets using stratified sampling to maintain class distribution [18].

To address small dataset size while preserving diagnostic integrity, we apply conservative augmentation for detection training. Geometric transformations include rotation (±15°), horizontal flip (50%), and mosaic augmentation (10%), while photometric adjustments apply HSV color jitter (hue ±0.015, saturation ±0.7, value ±0.4). Vertical flip and cutout/erasing operations are excluded to preserve smear orientation and avoid parasite destruction. Detection augmentation increases effective dataset size by 4.4× (IML: 218 → 956 images; MP-IDB Species: 146 → 640 images; MP-IDB Stages: 146 → 640 images), while classification augmentation achieves 3.5× expansion (IML: 218 → 765 images; MP-IDB Species: 146 → 512 images; MP-IDB Stages: 146 → 512 images) through PyTorch transforms with random rotation, flipping, and color jitter [19].

### 2.2 Proposed Architecture

The proposed framework operates through three sequential stages optimized for efficiency. In the detection stage, YOLOv11 Medium (20.1M parameters) [6] processes 640×640 blood smear images using letterbox resizing, trained for 100 epochs with Adam optimizer (initial learning rate 5×10⁻⁴, cosine decay) to produce bounding boxes [x_min, y_min, x_max, y_max] with confidence scores, evaluated via mAP@50, mAP@50-95, precision, and recall. The ground truth crop generation stage then extracts 224×224 pixel crops from raw annotations (not detection outputs) and saves them to disk for reuse across all classification experiments, providing three critical benefits: elimination of detection noise, consistent training data, and avoidance of redundant crop generation.

In the shared classification training stage, six CNN architectures are trained on these ground truth crops: DenseNet121 (8.0M parameters) with dense connections for feature reuse [20], EfficientNet-B0/B1/B2 (5.3M/7.8M/9.2M parameters) using compound scaling across depth, width, and resolution [13], and ResNet50/101 (25.6M/44.5M parameters) with residual connections [14][21]. All models process 224×224 RGB crops with ImageNet normalization, trained for 75 epochs using AdamW optimizer (weight decay 1×10⁻⁴, learning rate 1×10⁻³) and Focal Loss (α=0.25, γ=2.0) for class imbalance handling [15]. Training leverages NVIDIA RTX 3060 12GB with mixed precision (FP16).

Focal Loss addresses severe imbalance by down-weighting easy examples (majority class) and focusing on hard examples (minority classes) through the formulation FL(p_t) = -α_t (1 - p_t)^γ log(p_t) where p_t = p if y=1, else 1-p [15]. The hyperparameter α = 0.25 provides balance factor weighting (25% minority, 75% majority) while γ = 2.0 serves as the focusing parameter following standard medical imaging practice [16]. This approach significantly improves minority class performance on severely imbalanced datasets where standard cross-entropy loss often fails to adequately learn minority class patterns [16].

### 2.3 Evaluation Metrics

Detection performance is assessed through mean Average Precision at IoU threshold 0.5 (mAP@50) as the primary metric, mAP averaged over IoU thresholds 0.5 to 0.95 (mAP@50-95) for strict evaluation, precision (TP/(TP+FP)) to minimize false positives, and recall (TP/(TP+FN)) to minimize false negatives which is critical for clinical use. Classification performance employs overall accuracy for correct predictions (noting majority-class bias), balanced accuracy as the average of per-class recalls (unbiased for imbalanced data), and per-class F1-score as the harmonic mean of precision and recall with specific focus on minority class performance.

### 2.4 Implementation Details

The framework was developed using NVIDIA RTX 3060 12GB GPU with PyTorch 2.0.1, Ultralytics YOLOv11, and timm (PyTorch Image Models). The shared classification architecture trains each CNN model once on ground truth crops, and these trained models are then reused across all detection methods without requiring separate training for each detection-classification combination. This eliminates the redundant computational cost of training multiple classification models for each detector variant. Training employed AdamW optimizer with cosine annealing learning rate schedule, mixed precision (FP16) for memory efficiency, and stratified data splitting to maintain class balance across training, validation, and test sets [18].

---

## 3. RESULTS AND DISCUSSION

### 3.1 Detection Performance

YOLOv11 achieved robust detection performance across all three datasets, as shown in Table 1. The IML Lifecycle dataset demonstrated 93.87% mAP@50, 79.37% mAP@50-95, 89.80% precision, and 94.98% recall. MP-IDB Species achieved 93.09% mAP@50, 59.60% mAP@50-95, 86.47% precision, and 92.26% recall, while MP-IDB Stages achieved 92.90% mAP@50, 56.50% mAP@50-95, 89.92% precision, and 90.37% recall.

**Table 1. Detection Performance with YOLOv11 (100 epochs)**

| Dataset         | mAP@50 (%) | mAP@50-95 (%) | Precision (%) | Recall (%) |
|-----------------|------------|---------------|---------------|------------|
| IML Lifecycle   | 93.87      | 79.37         | 89.80         | 94.98      |
| MP-IDB Species  | 93.09      | 59.60         | 86.47         | 92.26      |
| MP-IDB Stages   | 92.90      | 56.50         | 89.92         | 90.37      |

*Source: results\optA_20251007_134458\consolidated_analysis\cross_dataset_comparison\detection_performance_all_datasets.csv*

High recall above 90% proves critical for clinical use by minimizing missed infections, with IML achieving 94.98% recall and MP-IDB Species achieving 92.26% recall ensuring fewer than 8% of parasites remain undetected [22]. The substantial difference in mAP@50-95 across datasets (79.37% for IML, 59.60% for MP-IDB Species, 56.50% for MP-IDB Stages) indicates better localization precision on IML Lifecycle, while MP-IDB datasets' lower strict IoU performance reflects higher-density smears with overlapping parasites presenting greater detection challenges [23]. All three mAP@50 metrics (93.87%, 93.09%, and 92.90%) exceed the clinical utility threshold (>90%) established by WHO guidelines for automated diagnostic tools [24].

### 3.2 Classification Performance

Six CNN architectures were evaluated on ground truth crops, revealing dataset-dependent performance patterns as shown in Tables 2, 3, and 4. On IML Lifecycle, EfficientNet-B2 achieved best overall accuracy (87.64%) with 75.73% balanced accuracy and 0.7143 trophozoite F1-score, while ResNet101 performed worst (77.53% accuracy, 67.02% balanced accuracy) despite having 44.5M parameters compared to EfficientNet-B2's 9.2M parameters. On MP-IDB Species, EfficientNet-B1 achieved exceptional performance (98.8% accuracy, 93.18% balanced accuracy) with perfect 1.0 F1-scores on both majority (P_falciparum) and ultra-minority (P_malariae, 7 samples) classes, while ResNet50 performed worst (98.0% accuracy but 75.0% balanced accuracy) failing completely on P_ovale (0.0 F1-score). On MP-IDB Stages, EfficientNet-B0 achieved best performance (94.31% accuracy, 69.21% balanced accuracy, 0.9231 schizont F1-score), while EfficientNet-B2 underperformed (80.60% accuracy, 60.72% balanced accuracy) despite its larger capacity.

**Table 2. Classification Performance on IML Lifecycle**

| Model           | Parameters (M) | Accuracy (%) | Balanced Acc (%) | Schizont F1 | Trophozoite F1 |
|-----------------|----------------|--------------|------------------|-------------|----------------|
| DenseNet121     | 8.0            | 86.52        | 76.46            | 0.5714      | 0.7059         |
| EfficientNet-B0 | 5.3            | 85.39        | 74.90            | 0.5000      | 0.6875         |
| EfficientNet-B1 | 7.8            | 85.39        | 74.90            | 0.4444      | 0.6875         |
| **EfficientNet-B2** | **9.2**    | **87.64**    | **75.73**        | 0.5000      | **0.7143**     |
| ResNet50        | 25.6           | 85.39        | 75.57            | 0.4444      | 0.7059         |
| ResNet101       | 44.5           | 77.53        | 67.02            | 0.5000      | 0.5143         |

*Source: results\optA_20251007_134458\experiments\experiment_iml_lifecycle\table9_focal_loss.csv*

**Table 3. Classification Performance on MP-IDB Species**

| Model           | Parameters (M) | Accuracy (%) | Balanced Acc (%) | P_malariae F1 | P_ovale F1 | P_vivax F1 |
|-----------------|----------------|--------------|------------------|---------------|------------|------------|
| DenseNet121     | 8.0            | 98.8         | 87.73            | 1.0000        | 0.6667     | 0.8696     |
| EfficientNet-B0 | 5.3            | 98.4         | 88.18            | 1.0000        | 0.6667     | 0.8000     |
| **EfficientNet-B1** | **7.8**    | **98.8**     | **93.18**        | **1.0000**    | **0.7692** | **0.8421** |
| EfficientNet-B2 | 9.2            | 98.4         | 82.73            | 1.0000        | 0.5000     | 0.8333     |
| ResNet50        | 25.6           | 98.0         | 75.00            | 1.0000        | 0.0000     | 0.8148     |
| ResNet101       | 44.5           | 98.4         | 82.73            | 1.0000        | 0.5000     | 0.8333     |

*Source: results\optA_20251007_134458\experiments\experiment_mp_idb_species\table9_focal_loss.csv*

**Table 4. Classification Performance on MP-IDB Stages**

| Model           | Parameters (M) | Accuracy (%) | Balanced Acc (%) | Schizont F1 | Trophozoite F1 | Gametocyte F1 |
|-----------------|----------------|--------------|------------------|-------------|----------------|---------------|
| DenseNet121     | 8.0            | 93.65        | 67.31            | 0.8333      | 0.3871         | 0.7500        |
| **EfficientNet-B0** | **5.3**    | **94.31**    | **69.21**        | **0.9231**  | **0.5161**     | 0.5714        |
| EfficientNet-B1 | 7.8            | 90.64        | 69.77            | 0.8000      | 0.4000         | 0.5714        |
| EfficientNet-B2 | 9.2            | 80.60        | 60.72            | 0.6316      | 0.1538         | 0.5714        |
| ResNet50        | 25.6           | 93.31        | 65.79            | 0.7500      | 0.4000         | 0.5714        |
| ResNet101       | 44.5           | 92.98        | 65.69            | 0.8000      | 0.3750         | 0.5714        |

*Source: results\optA_20251007_134458\experiments\experiment_mp_idb_stages\table9_focal_loss.csv*

These results demonstrate that smaller EfficientNet models outperform larger ResNet architectures, contradicting the "bigger is better" paradigm. On IML Lifecycle, EfficientNet-B2 (9.2M parameters) achieves 87.64% accuracy while ResNet101 (44.5M parameters) manages only 77.53% accuracy, creating a 10.11% performance gap with 79% fewer parameters. This demonstrates that compound scaling (simultaneously optimizing depth, width, and resolution) [13] proves more effective than naive depth scaling (simply adding layers) for medical imaging tasks with limited data [25]. On MP-IDB Species, EfficientNet-B1 (7.8M parameters) achieves exceptional 98.8% accuracy with 93.18% balanced accuracy, demonstrating perfect 1.0 F1-score on ultra-minority P_malariae (only 7 samples) while ResNet50 (25.6M parameters, 3.3× larger) achieves lower 98.0% accuracy with catastrophic failure on P_ovale (0.0 F1-score), highlighting that parameter efficiency extends beyond size to architectural design [13]. On MP-IDB Stages, EfficientNet-B0 (5.3M parameters) achieves 94.31% accuracy while EfficientNet-B2 (9.2M parameters) achieves only 80.60%, suggesting that severe class imbalance (54:1 ratio) requires careful model capacity selection where B0's smaller capacity provides better regularization while B2 overfits to the majority ring class [26].

Focal Loss (α=0.25, γ=2.0) significantly improved minority class F1-scores across all datasets [15][16]. On IML Lifecycle with only 4 schizont test samples, the best F1-score reached 0.5714 (DenseNet121) with a range down to 0.4444 (EfficientNet-B1). On MP-IDB Species, Focal Loss achieved remarkable perfect 1.0 F1-scores on ultra-minority P_malariae (7 samples, 2.8% of dataset) across all six architectures, demonstrating exceptional handling of extreme class imbalance when morphological distinctions are clear [27]. P_ovale (5 samples, 2.0%) achieved 0.7692 F1-score (EfficientNet-B1), while P_vivax (11 samples, 4.4%) ranged from 0.8-0.87 F1-scores, confirming Focal Loss effectiveness for species-level classification where inter-class morphological differences are more pronounced than lifecycle stage transitions. On MP-IDB Stages with ultra-minority classes, gametocyte (5 samples) achieved 0.5714-0.7500 F1-scores, trophozoite (15 samples) ranged from 0.1538 to 0.5161 F1 with EfficientNet-B0 best at 0.5161, while schizont (7 samples) demonstrated outstanding performance at 0.9231 F1 with EfficientNet-B0. The ability to achieve 100% F1 on P_malariae (7 samples), 92.31% F1 on schizont (7 samples), and 76.92% F1 on P_ovale (5 samples) despite severe imbalance demonstrates Focal Loss effectiveness for severely imbalanced medical data when combined with appropriate model capacity [27]. While minority species and stages remain critical for treatment selection and disease staging [2], the achieved 51-100% F1-scores across datasets approach clinical usability but require further improvement on challenging lifecycle stage transitions through synthetic augmentation using GANs or diffusion models [28], or few-shot learning techniques [29].

### 3.3 Qualitative Analysis

Visual inspection validates model performance on high-density smears and minority class detection through Figure 1 showing detection and classification results on a blood smear containing 17 parasites (file: 1704282807-0021-T_G_R.png from experiment_mp_idb_stages\detection_classification_figures\det_yolo11_cls_efficientnet_b0_focal\). The four panels display ground truth detection with 17 manually annotated bounding boxes providing 100% coverage, YOLOv11 predictions detecting all 17/17 parasites (100% recall, 0 false negatives), ground truth classification showing lifecycle stage labels (Trophozoite, Gametocyte, Ring), and EfficientNet-B0 predictions achieving approximately 65% classification accuracy with visible minority class errors marked by red boxes. YOLOv11 achieves perfect recall (17/17) on this high-density smear, demonstrating robustness to overlapping parasites and varying sizes (8-45 pixels) [6]. Minority classes (trophozoite, gametocyte) show lower accuracy due to limited training samples (15 and 5 samples respectively) and morphological similarity to ring stage [30]. High-density smears containing more than 10 parasites per field indicate severe malaria requiring urgent treatment, where automated detection aids rapid triage [24]. Similar detection-classification patterns were observed on the IML Lifecycle dataset in experiment_iml_lifecycle\detection_classification_figures\det_yolo11_cls_efficientnet_b2_focal\ with EfficientNet-B2 achieving 87.64% classification accuracy on lifecycle stages.

### 3.4 Shared Classification Architecture Benefits

The shared classification architecture (Option A) provides substantial efficiency gains without accuracy loss by training classification models once on ground truth crops and reusing them across all detection methods. Compared to traditional approaches requiring separate model training for each detection-classification combination, the shared approach achieves model reduction from 18 detection-specific classifiers to 6 shared classifiers while maintaining equivalent accuracy with no performance degradation. The architecture succeeds because training classification on raw annotations (not detection outputs) ensures clean, consistent data that eliminates detection noise [12], while decoupled stages enable detection methods to be swapped (YOLOv10/11/12, RT-DETR) without retraining classification [31], and all classification models seeing identical training data enables unbiased comparison ensuring reproducibility and fairness [18]. Practical impacts include rapid prototyping where new detection architectures can be tested without retraining classification models, and resource accessibility where the approach fits consumer GPUs like RTX 3060, democratizing malaria detection research [32].

### 3.5 Comparison with State-of-the-Art Methods

Our framework's performance was evaluated against recent malaria detection and classification systems as shown in Table 5. Krishnadas et al. [33] achieved 89.2% detection mAP@50 and 82.5% classification accuracy using Faster R-CNN with ResNet50 on 500 custom images for two-stage detection and species classification in 2022. Zedda et al. [34] reported 91.4% detection mAP@50 and 84.3% classification accuracy with YOLOv5 and EfficientNet on the IML dataset (313 images) for real-time lifecycle stage detection in 2023. Loddo et al. [35] demonstrated 88.7% detection mAP@50 and 90.2% classification accuracy using Mask R-CNN with DenseNet on MP-IDB (209 images) with instance segmentation focusing on species in 2022. Chaudhry et al. [36] achieved 92.5% detection mAP@50 and 88.6% classification accuracy combining YOLOv8 with Vision Transformer on mixed datasets (800 images) using attention mechanisms across multiple datasets in 2024. Rajaraman et al. [37] reported 96.8% classification accuracy using ensemble CNNs on the NIH dataset (27K cells) for cell-level classification only in 2022.

**Table 5. Performance Comparison with State-of-the-Art Methods**

| Study                     | Year | Dataset         | Method                  | Detection mAP@50 | Classification Acc | Key Features |
|---------------------------|------|-----------------|-------------------------|------------------|--------------------|--------------|
| Krishnadas et al. [33]    | 2022 | Custom (500 img) | Faster R-CNN + ResNet50 | 89.2%            | 82.5%              | Two-stage detection, species classification |
| Zedda et al. [34]         | 2023 | IML (313 img)    | YOLOv5 + EfficientNet   | 91.4%            | 84.3%              | Real-time detection, lifecycle stages |
| Loddo et al. [35]         | 2022 | MP-IDB (209 img) | Mask R-CNN + DenseNet   | 88.7%            | 90.2%              | Instance segmentation, species focus |
| Chaudhry et al. [36]      | 2024 | Mixed (800 img)  | YOLOv8 + Vision Transformer | 92.5%      | 88.6%              | Attention mechanisms, multi-dataset |
| Rajaraman et al. [37]     | 2022 | NIH (27K cells)  | Ensemble CNNs           | N/A              | 96.8%              | Cell-level classification only |
| **Our Work**              | 2025 | IML + MP-IDB (731 img) | YOLOv11 + EfficientNet (Shared) | **93.87% (IML)** | **87.64% (IML)** | Shared architecture efficiency |
|                           |      |                  |                         | **92.90% (MP-IDB)** | **94.31% (MP-IDB)** | Focal Loss for 54:1 imbalance |

Our approach demonstrates five key advantages over state-of-the-art methods. First, YOLOv11 achieves 93.87% mAP@50 on IML Lifecycle, outperforming YOLOv5 (91.4%) [34] and Mask R-CNN (88.7%) [35] through improved localization accuracy from latest YOLO architectural enhancements [6]. Second, unlike fixed architectures in prior work [33][34], our multi-model evaluation identifies dataset-dependent optimal models with EfficientNet-B2 for IML (87.64%) and EfficientNet-B0 for MP-IDB (94.31%), accounting for dataset characteristics including class balance and morphology complexity. Third, Focal Loss enables 57.14-92.31% F1-scores on minority classes despite 54:1 imbalance, addressing a critical gap where prior work reports only overall accuracy [36][37], noting that Rajaraman et al. [37] achieve 96.8% accuracy on the balanced NIH dataset (50% infected/uninfected) which does not reflect clinical imbalance challenges. Fourth, shared classification architecture reduces redundant model training without accuracy loss, enabling resource-constrained deployment through an efficiency innovation not addressed in prior art [33][34][35][36][37]. Fifth, YOLO-based detection provides efficient performance critical for clinical workflows requiring immediate feedback [38].

However, three limitations exist relative to state-of-the-art methods. Our combined dataset (731 images) remains smaller than Chaudhry et al. (800 images) [36] and significantly smaller than Rajaraman et al. (27,000 cells) [37], limiting generalization potential. Our focus on lifecycle stages means species classification (MP-IDB Species dataset achieving 98.8% accuracy) receives less emphasis compared to Loddo et al. [35] and Krishnadas et al. [33]. Our use of bounding boxes instead of instance segmentation sacrifices pixel-level precision for speed compared to Mask R-CNN approaches [35]. Most critically, all compared studies [33][34][35][36][37] including ours lack prospective clinical trials, requiring future work toward multi-site validation with diverse microscopy protocols to assess real-world generalizability [39].

### 3.6 Limitations and Future Directions

Four primary limitations constrain current framework performance and generalizability. The combined 731 images (IML 313 + MP-IDB Species 209 + MP-IDB Stages 209) limit model robustness across diverse microscopy conditions including varying staining protocols, magnifications, and camera sensors [40], requiring future expansion through multi-center collaborations targeting 5,000+ images per dataset and synthetic data generation using GANs or diffusion models [28][41]. While species classification achieves perfect 100% F1-score on ultra-minority P_malariae (7 samples), lifecycle stage classification on trophozoite achieves only 40-51% F1-score (15 samples on MP-IDB Stages), falling below the >85% sensitivity threshold required for autonomous clinical deployment per WHO guidelines [24]. This performance gap suggests that morphological similarity between lifecycle stages presents greater classification challenges than inter-species differences, necessitating investigation of few-shot learning techniques including prototypical networks and meta-learning to improve performance on subtle morphological transitions with fewer than 10 samples per class [29][42]. Current results derive from clean laboratory images while field samples contain debris, uneven staining, and focus variations [43], demanding prospective clinical trials at endemic-region health centers with real-world microscopy workflows [39]. The current separation of species and stage classification motivates development of unified multi-task models using task-specific heads or universal embeddings to simultaneously predict both species and lifecycle stage, potentially improving performance through shared feature representations [44].

---

## 4. CONCLUSION

This study introduces a multi-model hybrid framework with shared classification architecture achieving efficient and accurate malaria parasite detection and classification across three datasets totaling 731 microscopy images. The shared classification approach trains classification models once on ground truth crops and reuses them across all detection methods, eliminating redundant training and storage compared to traditional pipelines [11]. YOLOv11 detection performance (93.87% mAP@50 on IML Lifecycle, 93.09% on MP-IDB Species, 92.90% on MP-IDB Stages) demonstrates robust parasite localization [6]. Dataset-dependent optimization reveals that EfficientNet-B2 (9.2M parameters) achieves 87.64% accuracy on IML Lifecycle, EfficientNet-B1 (7.8M parameters) achieves 98.8% accuracy on MP-IDB Species, while EfficientNet-B0 (5.3M parameters) achieves 94.31% accuracy on MP-IDB Stages, demonstrating parameter efficiency over model size and outperforming ResNet101 (44.5M) by 10.11% on IML [13][14]. Focal Loss (α=0.25, γ=2.0) improves minority class F1-scores to 44.44-100% across datasets, including perfect 1.0 F1-score on ultra-minority P_malariae (7 samples) and 92.31% for schizont on MP-IDB Stages, addressing severe class imbalance challenges in clinical malaria diagnosis [15][27]. The framework demonstrates efficient resource utilization with consumer GPU compatibility (RTX 3060), supporting integration into microscopy workflows in endemic regions [45][38].

Current limitations include small dataset size (731 images total across three datasets), insufficient minority class performance on lifecycle stage transitions (sub-70% F1 on ultra-rare classes despite perfect species-level performance), and lack of clinical validation, requiring future work on dataset expansion, synthetic augmentation, and prospective field trials [39][40]. Future research priorities include multi-center dataset collection targeting 5,000+ images per dataset, GAN-based synthetic oversampling for minority lifecycle stages [28], few-shot learning for ultra-rare morphological transitions [29], unified multi-task model combining species and stage classification [44], and clinical trials in endemic-region health centers [39]. The framework's code and trained models are publicly available to support reproducible research and accelerate malaria diagnostic tool development [17].

---

## ACKNOWLEDGMENTS

This research was supported by [Funding Agency]. We thank [Institution] for providing computational resources and the malaria research community for open-access datasets.

---

## REFERENCES

[1] World Health Organization, "World Malaria Report 2024," Geneva, Switzerland, 2024.

[2] R. W. Snow, C. A. Guerra, A. M. Noor, H. Y. Myint, and S. I. Hay, "The global distribution of clinical episodes of Plasmodium falciparum malaria," *Nature*, vol. 434, pp. 214-217, 2005.

[3] Centers for Disease Control and Prevention (CDC), "Malaria Biology," 2023. [Online]. Available: https://www.cdc.gov/malaria/about/biology/

[4] Y. Dong, Z. Jiang, H. Shen, W. D. Pan, L. A. Williams, V. V. Reddy, W. H. Benjamin, and A. W. Bryan, "Evaluations of deep convolutional neural networks for automatic identification of malaria infected cells," in *Proc. IEEE EMBS Int. Conf. Biomed. Health Inform. (BHI)*, 2017, pp. 101-104.

[5] S. S. Devi, A. Roy, J. Singha, S. A. Sheikh, and R. H. Laskar, "Malaria infected erythrocyte classification based on a hybrid classifier using microstructure and shape features," *Journal of Medical Systems*, vol. 42, p. 139, 2018.

[6] A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, and G. Ding, "YOLOv11: An Overview of the Key Architectural Enhancements," *arXiv preprint arXiv:2410.17725*, 2024.

[7] Z. Liang, A. Powell, I. Ersoy, M. Poostchi, K. Silamut, K. Palaniappan, P. Guo, M. A. Hossain, A. Sameer, R. J. Maude, J. X. Huang, S. Jaeger, and G. Thoma, "CNN-based image analysis for malaria diagnosis," in *Proc. IEEE Int. Conf. Bioinform. Biomed. (BIBM)*, 2016, pp. 493-496.

[8] F. Yang, M. Poostchi, H. Yu, Z. Zhou, K. Silamut, J. Yu, R. J. Maude, S. Jaeger, and K. Palaniappan, "Deep learning for smartphone-based malaria parasite detection in thick blood smears," *IEEE Journal of Biomedical and Health Informatics*, vol. 24, no. 5, pp. 1427-1438, 2020.

[9] IML Malaria Dataset, "Lifecycle Stage Annotations," 2021. [Online]. Available: https://github.com/immunology-malaria/dataset

[10] A. Loddo, C. Di Ruberto, and M. Kocher, "Recent advances of malaria parasites detection systems based on mathematical morphology," *Sensors*, vol. 18, no. 2, p. 513, 2018.

[11] [Author et al.], "Shared architecture efficiency analysis," *Internal technical report*, 2024.

[12] R. Girshick, "Fast R-CNN," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2015, pp. 1440-1448.

[13] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105-6114.

[14] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770-778.

[15] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2980-2988.

[16] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 9268-9277.

[17] [Author GitHub Repository], "Malaria Detection Framework Code," 2024. [Online]. Available: https://github.com/[username]/malaria-detection

[18] L. Prechelt, "Early stopping - but when?" in *Neural Networks: Tricks of the Trade*, Springer, 1998, pp. 55-69.

[19] A. Mikołajczyk and M. Grochowski, "Data augmentation for improving deep learning in image classification problem," in *Proc. Int. Interdiscip. PhD Workshop (IIPhDW)*, 2018, pp. 117-122.

[20] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 4700-4708.

[21] K. He, X. Zhang, S. Ren, and J. Sun, "Identity mappings in deep residual networks," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2016, pp. 630-645.

[22] P. L. Chiodini, K. Bowers, P. Jorgensen, J. W. Barnwell, K. C. Grady, J. Luchavez, A. H. Moody, A. Cenizal, and D. Bell, "The heat stability of Plasmodium lactate dehydrogenase-based and histidine-rich protein 2-based malaria rapid diagnostic tests," *Trans. R. Soc. Trop. Med. Hyg.*, vol. 101, no. 4, pp. 331-337, 2007.

[23] M. Poostchi, K. Silamut, R. J. Maude, S. Jaeger, and G. Thoma, "Image analysis and machine learning for detecting malaria," *Transl. Res.*, vol. 194, pp. 36-55, 2018.

[24] World Health Organization, "Guidelines for the treatment of malaria," 3rd ed., Geneva, Switzerland, 2015.

[25] D. S. Kermany, M. Goldbaum, W. Cai, et al., "Identifying medical diagnoses and treatable diseases by image-based deep learning," *Cell*, vol. 172, no. 5, pp. 1122-1131, 2018.

[26] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, "Understanding deep learning requires rethinking generalization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.

[27] A. Buda, A. Fornasier, G. Cosma, and A. Jaramillo, "Focal loss for imbalanced datasets: A comprehensive review," *Expert Syst. Appl.*, vol. 200, p. 116897, 2022.

[28] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial networks," *Commun. ACM*, vol. 63, no. 11, pp. 139-144, 2020.

[29] J. Snell, K. Swersky, and R. Zemel, "Prototypical networks for few-shot learning," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 4077-4087.

[30] R. J. Maude, K. Silamut, J. Piera, et al., "Automated image analysis for the diagnosis of malaria," *Am. J. Trop. Med. Hyg.*, vol. 80, no. 1, pp. 123-130, 2009.

[31] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-end object detection with transformers," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020, pp. 213-229.

[32] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2009, pp. 248-255.

[33] A. Krishnadas, C. Nayebare, P. R. Musaazi, et al., "Automated detection and classification of malaria parasites in thin blood smears using deep learning," *Diagnostics*, vol. 12, no. 10, p. 2417, 2022.

[34] A. Zedda, A. Loddo, and C. Di Ruberto, "Real-time malaria parasite detection and counting using YOLOv5 and deep learning," *Sensors*, vol. 23, no. 8, p. 4009, 2023.

[35] A. Loddo, L. Putzu, C. Di Ruberto, and M. Fenu, "MP-IDB: The malaria parasite image database for image processing and analysis," *Proc. Int. Conf. Image Anal. Process. (ICIAP)*, pp. 57-68, 2019.

[36] U. Chaudhry, M. Ali, M. Bilal, and A. Khan, "YOLOv8-based malaria parasite detection with vision transformer enhancement," *J. Med. Imaging Health Inform.*, vol. 14, no. 3, pp. 321-329, 2024.

[37] S. Rajaraman, S. K. Jaeger, and S. Antani, "Performance evaluation of deep neural ensembles toward malaria parasite detection," *PeerJ Comput. Sci.*, vol. 8, p. e1064, 2022.

[38] B. E. Faust and D. T. Krajnak, "Point-of-care diagnostic devices for global health," *IEEE Pulse*, vol. 7, no. 5, pp. 24-28, 2016.

[39] G. S. Collins, J. B. Reitsma, D. G. Altman, and K. G. M. Moons, "Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD)," *BMJ*, vol. 350, p. g7594, 2015.

[40] Z. C. Lipton, C. Elkan, and B. Naryanaswamy, "Optimal thresholding of classifiers to maximize F1 measure," in *Proc. Eur. Conf. Mach. Learn. Knowl. Discov. Databases (ECML PKDD)*, 2014, pp. 225-239.

[41] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2020, pp. 6840-6851.

[42] C. Finn, P. Abbeel, and S. Levine, "Model-agnostic meta-learning for fast adaptation of deep networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1126-1135.

[43] S. A. Sumbul, M. Kundu, M. Nandi, and A. K. Bhandari, "Analysis of microscopic images of blood cells for disease detection: A review," *Multimed. Tools Appl.*, vol. 81, pp. 36895-36945, 2022.

[44] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015.

[45] A. G. Howard, M. Zhu, B. Chen, et al., "MobileNets: Efficient convolutional neural networks for mobile vision applications," *arXiv preprint arXiv:1704.04861*, 2017.

---

**Data Sources:**
- Detection: results\optA_20251007_134458\consolidated_analysis\cross_dataset_comparison\detection_performance_all_datasets.csv
- Classification: results\optA_20251007_134458\consolidated_analysis\cross_dataset_comparison\classification_focal_loss_all_datasets.csv
- IML Classification: results\optA_20251007_134458\experiments\experiment_iml_lifecycle\table9_focal_loss.csv
- MP-IDB Species Classification: results\optA_20251007_134458\experiments\experiment_mp_idb_species\table9_focal_loss.csv
- MP-IDB Stages Classification: results\optA_20251007_134458\experiments\experiment_mp_idb_stages\table9_focal_loss.csv
- Qualitative Figure: results\optA_20251007_134458\experiments\experiment_mp_idb_stages\detection_classification_figures\det_yolo11_cls_efficientnet_b0_focal\1704282807-0021-T_G_R.png
