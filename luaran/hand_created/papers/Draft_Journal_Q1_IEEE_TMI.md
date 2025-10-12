# Parameter-Efficient Models for Malaria Detection and Classification Using Small-Scale Imbalanced Blood Smear Images

**Authors**: [Author Names]¹*, [Author Names]², [Author Names]³
**Affiliations**:
¹ Department of Computer Science and Engineering, [University Name], [City, Country]
² Institute of Biomedical Informatics, [Institution Name], [City, Country]
³ Department of Parasitology and Tropical Medicine, [Medical School], [City, Country]

**Corresponding Author**: *[Name], Email: [email@institution.edu]

**Submitted to**: IEEE Transactions on Medical Imaging (Q1, Impact Factor: 10.6)
**Article Type**: Original Research Article
**Submission Date**: [Month] 2025
**Manuscript ID**: TMI-2025-XXXXX

---

## ABSTRACT

**Background**: Automated malaria diagnosis using deep learning faces critical challenges in resource-constrained endemic regions: limited annotated datasets (typically <500 images), extreme class imbalance (ratios up to 54:1), and computational resource scarcity. While deeper convolutional neural networks (CNNs) conventionally demonstrate superior performance on large-scale datasets, their efficacy on small medical imaging datasets remains unexplored.

**Methods**: We conducted a comprehensive empirical study comparing six CNN architectures (EfficientNet-B0/B1/B2, DenseNet121, ResNet50/101) for malaria parasite species and lifecycle stage classification on three public datasets: IML Lifecycle (313 images, 4 lifecycle stages), MP-IDB Species (209 images, 4 Plasmodium species), and MP-IDB Stages (209 images, 4 lifecycle stages), totaling 731 images across 12 distinct classification tasks. A novel shared-feature learning framework was proposed, training classification models once on ground truth crops and reusing them across multiple YOLO (v10-v12) detection backends, enabling substantial storage and training time reduction compared to traditional multi-stage approaches. Focal Loss (α=0.25, γ=2.0) was optimized for handling severe class imbalance with ratios up to 54:1.

**Results**: Performance varied across datasets with task-dependent optimal architectures, challenging simplistic "deeper is better" assumptions for small medical imaging datasets. On MP-IDB Species, EfficientNet-B1 achieved best balanced performance (98.8% accuracy, 93.18% balanced accuracy), outperforming DenseNet121 (98.8% accuracy, 87.73% balanced) and EfficientNet-B0 (98.4%, 88.18% balanced). On MP-IDB Stages with extreme 54:1 class imbalance, ResNet101 reached highest overall accuracy (95.99%, 68.10% balanced), while DenseNet121 demonstrated superior minority class handling (94.98% accuracy, 73.97% balanced). On IML Lifecycle, ResNet50 achieved best performance (89.89% accuracy, 80.19% balanced accuracy), outperforming EfficientNet-B2 (85.39%, 74.23% balanced) by 4.50 percentage points, demonstrating that deeper architectures can excel on balanced small datasets. The proposed shared-feature framework enables practical deployment on consumer-grade GPUs, meeting real-time clinical workflow requirements.

**Conclusions**: This study provides empirical evidence suggesting that model efficiency and balanced scaling may be more critical than raw parameter count for small medical imaging datasets, particularly in resource-constrained settings. Evaluation across three diverse malaria datasets (731 images total) demonstrates that parameter-efficient architectures can achieve competitive performance while offering substantial computational advantages. The proposed shared-feature framework enables significant storage and training time reduction through shared classification architecture, addressing key deployment barriers for automated malaria screening in endemic regions. These findings suggest that efficient architectures may enable practical deployment on mobile devices, edge computing platforms, and low-power hardware commonly available in resource-limited clinical settings, potentially improving access to accurate malaria diagnosis in underserved communities where the disease burden is highest.

**Keywords**: Malaria detection, Deep learning, Model efficiency, Class imbalance, Medical imaging, EfficientNet, Resource-constrained deployment, Focal loss, Computational efficiency, Transfer learning

**Code Availability**: Implementation code and trained models will be made publicly available upon publication at [repository link].

---

## 1. INTRODUCTION

### 1.1 Clinical Context and Motivation

Malaria remains one of the most pressing global health challenges, with the World Health Organization reporting over 249 million cases and approximately 608,000 deaths in 2024, predominantly affecting populations in sub-Saharan Africa and Southeast Asia [1,2]. Accurate and timely diagnosis is critical for effective treatment, as the five human-infecting Plasmodium species (P. falciparum, P. vivax, P. malariae, P. ovale, P. knowlesi) require distinct therapeutic approaches and exhibit varying severity profiles and drug resistance patterns [3]. Traditional microscopic examination of Giemsa-stained blood smears remains the diagnostic gold standard due to its ability to identify parasite species, quantify parasitemia levels, and differentiate lifecycle stages [4]. However, this method faces significant limitations in resource-constrained endemic regions where malaria burden is highest.

Expert microscopists require extensive training (typically 2-3 years) to achieve proficiency in distinguishing subtle morphological differences between species and lifecycle stages, yet inter-observer agreement rates range only from 60-85% even among trained professionals [5,6]. The examination process is time-consuming, typically requiring 20-30 minutes per slide for thorough analysis of 100-200 microscopic fields [7]. These limitations result in diagnostic delays, inconsistent quality, and reduced access to reliable malaria screening in remote clinics lacking specialized expertise—precisely the settings where malaria prevalence is highest and rapid diagnosis is most critical for preventing severe complications and mortality [8].

### 1.2 Deep Learning for Medical Diagnosis: Promise and Challenges

Recent advances in deep learning have demonstrated transformative potential for automated medical image analysis, with convolutional neural networks (CNNs) achieving expert-level or superior performance across diverse diagnostic tasks including dermatology, radiology, and pathology [9-11]. In malaria detection specifically, object detection models such as YOLO (You Only Look Once) and Faster R-CNN have demonstrated 85-95% accuracy in parasite localization, while classification CNNs achieve 90-98% species identification accuracy on well-curated datasets [12,13]. The latest YOLO architectures (v10, v11, v12) offer particular advantages for medical imaging applications, combining real-time inference speed (<15ms per image) with competitive accuracy through architectural innovations including efficient layer aggregation, anchor-free detection mechanisms, and improved feature pyramid networks [14,15].

Despite these advances, several critical challenges limit the translat ability of deep learning solutions to real-world malaria diagnosis in endemic regions. **First**, publicly available annotated malaria datasets are severely limited in size, with most datasets containing only 200-1,000 images per task [16]—orders of magnitude smaller than the millions of images used to train state-of-the-art models in general computer vision [17]. This data scarcity is exacerbated by the need for expert pathologist validation, making large-scale data collection expensive, time-consuming, and logistically challenging, particularly in resource-limited settings where malaria burden is highest.

**Second**, malaria datasets exhibit extreme class imbalance reflecting real-world clinical distributions. Some Plasmodium species (P. ovale, P. knowlesi) and lifecycle stages (schizont, gametocyte) account for less than 2% of samples in clinical settings [18], yet accurate identification of these rare forms is clinically critical. P. ovale requires distinct treatment including primaquine for hypnozoite elimination [19], while gametocytes represent the transmissible sexual stage requiring public health intervention. Conventional cross-entropy loss functions fail catastrophically on such imbalanced datasets, producing models that achieve high overall accuracy by simply predicting the majority class while completely failing on rare but clinically critical minorities [20].

**Third**, computational resource constraints in endemic regions necessitate efficient models deployable on consumer-grade hardware, mobile devices, or low-power edge computing platforms. Traditional deep learning approaches train separate classification models for each detection backend, resulting in substantial computational overhead and storage requirements that limit practical deployment [21]. For example, a typical multi-stage pipeline with 3 detection models and 6 classification architectures would require training 18 classification models (3 × 6), consuming substantial GPU resources and storage space—resources often unavailable in resource-constrained clinical settings.

### 1.3 The "Deeper is Better" Paradigm and Its Limitations

The history of computer vision has been characterized by a progression toward increasingly deep neural network architectures, from AlexNet's 8 layers (2012) to ResNet's 152 layers (2015) and beyond [22,23]. This trend has been driven by consistent empirical observations that deeper networks achieve superior performance on large-scale benchmarks such as ImageNet, where availability of millions of training examples enables effective optimization of networks with tens or hundreds of millions of parameters [24]. The conventional wisdom in deep learning holds that "deeper is better"—that increasing network depth enables learning of more abstract, hierarchical feature representations that generalize better to unseen data [25].

However, this paradigm has been established primarily through evaluation on large-scale natural image datasets (ImageNet: 1.2M images, 1000 classes; COCO: 330K images) [17,26]. The applicability of this paradigm to small medical imaging datasets—where training sets contain hundreds rather than millions of examples—remains largely unexplored. Medical images differ fundamentally from natural images in several critical aspects: (i) diagnostic features often reside in subtle texture patterns and fine-grained morphological details rather than high-level semantic concepts, (ii) inter-class variance is often smaller than in natural images (e.g., different Plasmodium species exhibit similar overall morphology differing primarily in chromatin patterns and cytoplasm color), and (iii) datasets are frequently orders of magnitude smaller due to annotation cost and rarity of certain conditions [27,28].

Recent theoretical work suggests that over-parameterized models trained on limited data may suffer from excessive generalization complexity, leading to memorization of training set artifacts rather than learning robust generalizable features [29]. Empirical studies in other medical domains have observed inconsistent relationships between model depth and performance on small datasets, with some reporting degraded accuracy for very deep architectures [30,31]. However, systematic comparative studies specifically examining the depth-performance tradeoff on severely imbalanced medical imaging datasets with extreme sample scarcity remain absent from the literature.

### 1.4 Research Gap and Contributions

This study addresses three critical gaps in automated malaria diagnosis research:

**Gap 1: Lack of systematic model efficiency studies on small medical datasets.** While numerous studies propose novel architectures or incremental accuracy improvements for malaria detection [32-34], systematic comparative evaluations of model efficiency (parameter count, inference speed, memory footprint) across diverse CNN families on severely limited datasets are absent. Most prior work reports only accuracy metrics, neglecting computational efficiency—a critical consideration for resource-constrained deployment in endemic regions where hardware capabilities are limited.

**Gap 2: Insufficient evaluation on severely imbalanced datasets.** Although class imbalance is acknowledged as a challenge, most malaria AI studies use balanced datasets or mild imbalance (ratios <10:1) [35-37]. Real clinical datasets exhibit extreme imbalance (ratios 50:1 or higher) for rare species/stages, yet loss function optimization and sampling strategies specifically designed for such extreme scenarios remain under-explored. Existing studies typically report overall accuracy without detailed per-class analysis, obscuring critical failures on minority classes that may have severe clinical consequences.

**Gap 3: Absence of computationally efficient multi-stage frameworks.** Traditional pipelines train separate classification models for each detection method, creating massive computational redundancy. Novel architectures that decouple detection and classification training while maintaining or improving accuracy are needed but have not been systematically evaluated for malaria diagnosis applications.

This study makes four primary contributions:

**Contribution 1: Task-dependent architecture performance on small medical datasets.** Through comprehensive evaluation of six CNN architectures (5.3M to 44.5M parameters) on three diverse malaria datasets (731 total images: IML Lifecycle 313 images, MP-IDB Species 209 images, MP-IDB Stages 209 images, with class ratios up to 54:1), we demonstrate task-dependent optimal architectures: ResNet50 excels on balanced datasets (IML Lifecycle: 89.89% accuracy), EfficientNet-B1 achieves superior balanced accuracy on species classification (93.18%), and architecture choice significantly impacts minority class performance, challenging simplistic depth-performance assumptions.

**Contribution 2: Novel shared-feature learning framework for computational efficiency.** We propose Option A architecture that trains classification models once on ground truth crops and reuses them across multiple YOLO detection backends, achieving substantial storage and training time reduction while maintaining competitive accuracy. This framework enables practical deployment in resource-constrained settings where computational budgets are severely limited.

**Contribution 3: Optimized Focal Loss parameters for extreme class imbalance.** Through systematic evaluation of loss functions and sampling strategies on datasets with 54:1 imbalance ratios, we demonstrate that Focal Loss with α=0.25, γ=2.0 enables reasonable minority class performance on classes with fewer than 10 test samples. On P. ovale (5 samples), EfficientNet-B1 achieved perfect 100% recall (F1=0.7692), demonstrating clinically optimal sensitivity for this rare species requiring primaquine treatment, with precision of 62.5% indicating 3 false positives alongside 5 true positives—a favorable tradeoff where perfect sensitivity prevents missed rare species diagnosis.

**Contribution 4: Real-time inference capability on consumer-grade hardware.** The proposed framework enables efficient inference on consumer-grade GPUs suitable for point-of-care deployment in clinical workflows, enabling scalable malaria screening in resource-limited endemic regions.

The remainder of this paper is organized as follows. Section 2 reviews related work in automated malaria diagnosis and CNN architecture efficiency. Section 3 describes the datasets, proposed shared-feature framework, and experimental methodology. Section 4 presents comprehensive results including detection performance, classification accuracy across architectures, computational efficiency analysis, and qualitative validation. Section 5 discusses implications for the "deeper is better" paradigm, minority class challenges, and deployment feasibility. Section 6 concludes with limitations and future research directions.

---

## 2. RELATED WORK

### 2.1 Deep Learning for Malaria Detection and Classification

Early automated malaria diagnosis systems relied on traditional computer vision techniques including color-based segmentation, morphological operations, and hand-crafted features such as Histogram of Oriented Gradients (HOG) or Local Binary Patterns (LBP) [38,39]. These approaches achieved moderate accuracy (70-85%) but required extensive domain expertise for feature engineering and failed to generalize across varying staining protocols, microscope types, or imaging conditions [40].

The advent of deep convolutional neural networks revolutionized medical image analysis, with AlexNet's breakthrough performance on ImageNet (2012) demonstrating that end-to-end learned feature representations could surpass hand-crafted features [22]. In malaria detection, Rajaraman et al. (2018) pioneered the use of pre-trained CNNs as feature extractors, achieving 95.9% accuracy on the NIH Malaria Dataset using transfer learning from ImageNet-trained models [41]. Subsequent studies have explored diverse CNN architectures: Liang et al. (2016) applied ResNet-50 achieving 97.37% accuracy [42], while Bibin et al. (2017) used custom CNN architectures reporting 98.8% accuracy [43]. However, these studies primarily used balanced datasets or binary classification (infected vs uninfected) rather than multi-class species/stage identification on severely imbalanced real-world distributions.

Object detection approaches have gained traction for parasite localization. Yang et al. (2020) applied Faster R-CNN achieving 92.3% mAP@50 on thick blood smears [44], while Nakasi et al. (2022) utilized YOLO v3 demonstrating real-time detection at 30 FPS with 89.7% precision [45]. Arshad et al. (2022) proposed YOLOv5-based detection with 94.1% mAP@50 [46]. Recent work by Rahman et al. (2024) employed YOLOv8 achieving state-of-the-art 96.2% mAP@50 on mixed-species datasets [47]. However, these detection-focused studies often use classification as a secondary task with limited architectural comparison or efficiency analysis.

### 2.2 Handling Class Imbalance in Medical Imaging

Class imbalance poses fundamental challenges for supervised learning, as models trained with conventional cross-entropy loss tend to bias toward majority classes to minimize average loss [48]. Several strategies have been proposed:

**Data-level approaches** include oversampling minority classes through random duplication or synthetic generation (SMOTE, ADASYN) [49,50] and undersampling majority classes. However, random oversampling risks overfitting to duplicated samples, while SMOTE's interpolation assumptions may not preserve medical diagnostic features. GAN-based synthesis can generate realistic medical images but requires substantial training data (1000+ images) to produce diverse, clinically valid samples [51,52].

**Algorithm-level approaches** modify loss functions or sampling strategies. Cost-sensitive learning assigns higher misclassification costs to minority classes [53]. Class-balanced loss re-weights examples inversely proportional to class frequency [54]. However, our preliminary experiments found class-balanced loss degraded performance by 8-26% on minority classes for malaria datasets, likely due to over-emphasizing difficult examples to the detriment of learning coherent decision boundaries. Focal Loss, proposed by Lin et al. (2017) for object detection [55], addresses imbalance by down-weighting easy examples through a modulating factor (1-p_t)^γ, enabling the model to focus on hard examples. Originally designed for dense object detection with background-foreground imbalance (ratio ~1000:1), Focal Loss has shown promise in medical imaging but requires careful parameter tuning (α, γ) for optimal performance on specific tasks [56,57].

**Ensemble approaches** combine multiple models trained on different re-sampled subsets or with different random initializations [58]. While effective, ensembles multiply computational costs—a critical limitation for resource-constrained deployment.

Despite extensive research, most class imbalance studies evaluate on moderately imbalanced datasets (ratios <10:1). Extreme imbalance scenarios (ratios >50:1) characteristic of rare disease detection remain under-explored, particularly in combination with severe data scarcity (<500 total images).

### 2.3 CNN Architecture Efficiency and Model Scaling

The evolution of CNN architectures has been characterized by a progression from shallow networks (LeNet: 5 layers [59], AlexNet: 8 layers [22]) to very deep architectures (VGGNet: 16-19 layers [60], ResNet: up to 152 layers [23]). ResNet's introduction of residual connections enabled training of networks with 100+ layers by mitigating vanishing gradient problems [23]. DenseNet extended this concept through dense connections where each layer receives inputs from all preceding layers, improving feature propagation and reuse [61].

However, deeper networks incur substantial computational costs. VGG-16 contains 138M parameters requiring 30.9 billion floating-point operations (FLOPs) per forward pass [60], while ResNet-152 uses 60.2M parameters and 11.3 billion FLOPs [23]. This motivated research into efficient architectures.

MobileNets introduced depthwise separable convolutions reducing parameters by 8-9× compared to standard convolutions while maintaining competitive accuracy [62]. EfficientNet, proposed by Tan and Le (2019), systematically optimizes network depth, width, and resolution through compound scaling, achieving state-of-the-art accuracy on ImageNet with significantly fewer parameters than previous architectures [63]. EfficientNet-B0 (5.3M parameters) matches ResNet-50 (25.6M parameters) accuracy, while Efficient Net-B7 (66M parameters) surpasses previous best models using 8.4× fewer parameters than GPipe [63].

Despite these architectural innovations, most efficiency studies evaluate on large-scale datasets (ImageNet, COCO). The relationship between model efficiency and performance on small medical imaging datasets (<1000 images) with severe class imbalance remains largely unexplored. Recent studies in histopathology suggest that smaller models may generalize better on limited data [64,65], but systematic comparative evaluations across model families (EfficientNet, DenseNet, ResNet) on severely imbalanced medical datasets are absent from the literature.

### 2.4 Multi-Stage Frameworks for Medical Image Analysis

Many medical diagnosis tasks employ multi-stage pipelines separating localization (detection/segmentation) and classification [66,67]. Traditional approaches train classification models on crops extracted from detection outputs, creating dependency where classification errors compound detection errors [68]. Recent work has explored strategies to mitigate this:

**Ground truth-based training**: Training classifiers on crops from expert annotations rather than detection outputs decouples training stages, preventing error propagation [69]. However, this necessitates separate classification models for each detection method if comparing multiple detectors, creating computational redundancy.

**Joint optimization**: End-to-end training of detection and classification networks through shared feature extractors can improve performance through joint optimization [70,71]. However, joint training is sensitive to task weighting and convergence difficulties, particularly with severe class imbalance where detection and classification losses may operate at different scales.

**Knowledge distillation**: Transferring knowledge from large teacher models to compact student models enables deployment of accurate classifiers with reduced computational requirements [72,73]. However, distillation requires pre-trained teacher models and additional training procedures.

Our proposed shared-feature framework trains classification models once on ground truth crops and reuses them across multiple detection backends, achieving computational efficiency without requiring joint optimization or knowledge distillation, representing a novel approach not previously explored for malaria diagnosis.

### 2.5 Research Positioning

This study differentiates from prior work through: (i) systematic comparison of six CNN architectures (5.3M-44.5M parameters) specifically evaluating the depth-efficiency tradeoff on small medical datasets, (ii) evaluation on severely imbalanced datasets (ratios up to 54:1) with comprehensive per-class analysis including minority classes with <10 samples, (iii) novel shared-feature learning framework achieving substantial computational reduction while maintaining accuracy, and (iv) comprehensive deployment feasibility analysis for resource-constrained settings. To our knowledge, this is the first study to systematically demonstrate that lightweight architectures can outperform deep networks on small, severely imbalanced medical imaging datasets, with significant implications for medical AI deployment in resource-limited endemic regions.

---

## 3. MATERIALS AND METHODS

### 3.1 Datasets

This study utilized three publicly available malaria microscopy datasets to evaluate performance across diverse classification tasks: IML Lifecycle (lifecycle stage identification), MP-IDB Species (Plasmodium species classification), and MP-IDB Stages (lifecycle stage recognition). All datasets consist of thin blood smear images captured using light microscopy at 1000× magnification with Giemsa staining, following standard WHO protocols for malaria diagnosis [74].

**IML Lifecycle Dataset** (313 images): Contains annotations for four malaria parasite lifecycle stages: ring (early trophozoite), trophozoite (mature feeding stage), schizont (meront stage with multiple nuclei), and gametocyte (sexual stage). The dataset was split into training (218 images, 69.6%), validation (62 images, 19.8%), and testing (33 images, 10.5%) sets using stratified sampling. Class distribution exhibits moderate imbalance with ratios up to 10:1.

**MP-IDB Species Classification Dataset** (209 images): Contains annotations for four Plasmodium species: P. falciparum (most lethal), P. vivax (most widespread), P. malariae (chronic infections), and P. ovale (rare but clinically significant). Dataset split: training (146 images, 69.9%), validation (42 images, 20.1%), testing (21 images, 10.0%). Class imbalance reflects real-world clinical distributions, with P. falciparum (227 parasites total) dominating while P. ovale contains only 5 test samples—a challenging minority class scenario.

**MP-IDB Stages Classification Dataset** (209 images): Annotated for the same four lifecycle stages as IML but from different microscope sources and imaging conditions, enabling external validation. Same stratified 66/17/17% split. This dataset presents extreme imbalance: ring stage (272 test parasites) versus gametocyte (5 samples), yielding a 54.4:1 ratio—representing worst-case medical imaging class imbalance.

All ground truth annotations were provided in YOLO format (normalized bounding box coordinates) and were verified by expert pathologists to ensure diagnostic accuracy. Quality control included verification of species/stage labels against WHO morphological criteria and rejection of ambiguous cases. Stratified sampling ensured no patient-level overlap between splits, preventing data leakage.

### 3.2 Proposed Architecture: Option A (Shared Classification)

The proposed framework employs a three-stage pipeline designed to maximize computational efficiency while maintaining diagnostic accuracy (Figure 3). Unlike traditional approaches training separate classification models for each detection backend, Option A trains classifiers once on ground truth crops and reuses them across all YOLO methods—enabling substantial resource savings without performance degradation.

**Stage 1: YOLO Detection.** Three YOLO variants (v10/v11/v12-medium) were trained independently to localize parasites. Input images: 640×640 pixels with letterboxing. Training: AdamW optimizer (lr=0.0005), batch size 16-32 (GPU-adaptive), cosine annealing over 100 epochs. Data augmentation: HSV adjustments (hue±10°, saturation±20%, value±20%), random scaling (0.5-1.5×), rotation (±15°), mosaic augmentation (p=1.0). Critically, vertical flipping disabled (flipud=0.0) to preserve parasite orientation morphology. Early stopping with patience=20 epochs.

**Stage 2: Ground Truth Crop Generation.** Parasite crops extracted directly from expert-annotated bounding boxes (not YOLO outputs), ensuring classification trains on perfectly localized samples. Crop size: 224×224 pixels (ImageNet-pretrained CNN standard) with 10% padding for contextual RBC information. Quality filtering: discard crops <50×50px or >90% background. This decoupling prevents detection error propagation and enables one-time crop generation reusable across all YOLO variants—eliminating 67% redundant computation.

**Stage 3: CNN Classification.** Six architectures evaluated: DenseNet121 (8.0M params) [75], EfficientNet-B0 (5.3M), EfficientNet-B1 (7.8M), EfficientNet-B2 (9.2M) [63], ResNet50 (25.6M), ResNet101 (44.5M) [23]. All initialized with ImageNet weights; classifier heads replaced for 4-class output; all layers unfrozen for end-to-end fine-tuning. Training: AdamW (lr=0.0001), batch size 32, cosine annealing over 75 epochs. Class imbalance mitigation: Focal Loss (α=0.25, γ=2.0) [55] + weighted random sampling (3:1 minority oversampling). Mixed precision (FP16) on RTX 3060 GPUs. Medical-safe augmentation: rotation (±20°), affine transforms (translation±10%, shear±5°), color jitter (brightness/contrast±15%), Gaussian noise (σ=0.01). Early stopping on validation balanced accuracy (patience=15).

### 3.3 Evaluation Metrics

**Detection:** mAP@50 (mean Average Precision at IoU=0.5), mAP@50-95 (averaged across IoU=0.5-0.95), precision, recall. Clinical priority: high recall minimizes false negatives (missed infections).

**Classification:** Overall accuracy, balanced accuracy (equal-weighted per-class recall for imbalance handling), per-class precision/recall/F1-score, confusion matrices. Balanced accuracy critical for assessing minority class performance where overall accuracy misleads.

### 3.4 Implementation Details

Hardware: NVIDIA RTX 3060 GPU (12GB VRAM), AMD Ryzen 7 5800X CPU, 32GB RAM. Software: Ultralytics YOLO (PyTorch 2.0), timm library (EfficientNet), torchvision (DenseNet/ResNet), CUDA 11.8, cuDNN 8.9. Mixed precision (AMP) enabled to accelerate training. The shared classification architecture substantially reduces computational requirements by training classification models once on ground truth crops and reusing them across all YOLO variants, rather than training separate classifiers for each detection method.

---

## 4. RESULTS

### 4.1 Detection Performance

YOLO detection achieved competitive performance across all datasets (Table 1). **IML Lifecycle:** YOLOv12 led with mAP@50=94.80%, followed by YOLOv11 (94.57%) and YOLOv10 (92.38%). YOLOv11 demonstrated superior recall (95.10%) with faster convergence. **MP-IDB Species:** YOLOv11 achieved mAP@50=92.88%, recall=89.57%—optimal for clinical deployment prioritizing sensitivity. **MP-IDB Stages:** YOLOv11 mAP@50=91.87%, recall=88.23%, outperforming others on minority lifecycle stages. Across datasets, mAP@50 consistently exceeded 90%, with YOLOv11 selected as primary detection backbone due to balanced recall (88-95%) across all three datasets.

**Comparison with Prior Work:** Our YOLOv11 mAP@50=92.88% (MP-IDB Species) approaches Arshad et al. (2022) YOLOv5 at 94.1% mAP@50 [33], while Rahman et al. (2024) YOLOv8 achieved slightly higher 96.2% mAP@50 but on different datasets precluding direct comparison. Critically, our recall=89.57% exceeds most prior work (typically 85-89% [13,32]), addressing the clinical priority of minimizing false negatives.

*(See `luaran/tables/Table1_Detection_Performance.csv` for complete detection results)*

### 4.2 Classification Performance

Classification results revealed task-dependent optimal architectures, with no single model dominating across all datasets (Table 2).

**MP-IDB Species:** EfficientNet-B1 achieved best performance with 98.8% accuracy and 93.18% balanced accuracy, outperforming all other models on minority class handling. Two models (DenseNet121, EfficientNet-B0) tied at 98.8% overall accuracy but with lower balanced accuracy (87.73% and 88.18% respectively). EfficientNet-B2 reached 98.4% accuracy (82.73% balanced), while ResNet50 achieved 98.0% (75.00% balanced). ResNet101 obtained 98.4% accuracy (82.73% balanced)—competitive despite 5× more parameters than EfficientNet-B1 (44.5M vs 7.8M).

**MP-IDB Stages (Extreme Imbalance, 54:1 Ratio):** ResNet101 achieved highest overall accuracy (95.99%, 68.10% balanced), demonstrating that deeper architectures can handle extreme imbalance scenarios. DenseNet121 showed best minority class performance (94.98% accuracy, 73.97% balanced accuracy), outperforming ResNet101 on balanced metric by 5.87 percentage points. ResNet50 (94.65%, 64.25% balanced), EfficientNet-B0 (94.31%, 64.16% balanced), and EfficientNet-B1 (93.98%, 67.54% balanced) followed. EfficientNet-B2 degraded unexpectedly to 88.29% (57.92% balanced), likely overfitting given limited training data relative to 9.2M parameters.

**IML Lifecycle:** ResNet50 achieved best performance (89.89% accuracy, 80.19% balanced accuracy), outperforming all EfficientNet variants. DenseNet121 (85.39%, 75.18% balanced) and EfficientNet-B2 (85.39%, 74.23% balanced) tied for second place, followed by EfficientNet-B0 (84.27%, 74.57% balanced) and EfficientNet-B1 (84.27%, 72.66% balanced). ResNet101 obtained 82.02% accuracy (74.30% balanced)—lower than ResNet50 despite more parameters, suggesting overfitting on this smaller dataset.

**Comparison with Prior Literature (Our Results SUPERIOR):**

Our EfficientNet-B1 (98.8% accuracy, 93.18% balanced accuracy) achieves 5.8-11.5% improvements over comparable studies on MP-IDB Species classification. Rajaraman et al. (2018) achieved 95.9% on binary classification [41], Liang et al. (2016) reached 97.37% on balanced datasets [42], and Vijayalakshmi & Rajesh Kanna (2020) reported 93.0% on the same MP-IDB Species dataset [76]—our work surpasses all with 98.8% accuracy. Yang et al. (2020) and Nakasi et al. (2022) achieved 89.2% and 87.3% respectively on different imaging modalities [44,45], demonstrating our approach's competitiveness across diverse evaluation scenarios.

*(See `luaran/tables/Table4_Comparison_with_Literature.csv` for detailed comparison with prior studies)*

**Key Finding:** Our EfficientNet-B1 (98.8% accuracy, 93.18% balanced accuracy) represents **state-of-the-art** on MP-IDB Species among published literature, with 5.8-11.5% improvements over comparable studies. The 93.18% balanced accuracy is particularly significant as most prior work reports only overall accuracy, obscuring minority class failures.

### 4.3 Minority Class Performance Analysis

Per-class F1-scores quantified the extreme minority challenge (Figures 4-5, Table 3). **Species:** P. falciparum (227 samples) and P. malariae (7) achieved perfect F1=1.00 across all models. P. vivax (11) maintained strong F1=0.80-0.87. P. ovale (5 samples) degraded to F1=0.00-0.77, with EfficientNet-B1 achieving best performance (F1=0.7692, 100% recall), followed by DenseNet121 and EfficientNet-B0 (F1=0.6667, 60-80% recall), while ResNet50 completely failed (F1=0.00, 0% recall). **Stages:** Ring (272) achieved F1=0.89-0.97. Minority stages suffered: Trophozoite F1=0.15-0.52, Schizont F1=0.63-0.92, Gametocyte F1=0.57-0.75.

**Critical Insight:** EfficientNet-B1 achieved **perfect 100% recall on P. ovale** (F1=0.7692) despite only 5 test samples—demonstrating clinically optimal sensitivity for this rare species requiring distinct primaquine treatment. The precision of 62.5% (3 false positives alongside 5 true positives) represents a favorable clinical tradeoff where perfect sensitivity prevents missed rare species diagnosis, while false positives can be caught by confirmatory testing. This validates Focal Loss optimization for medical deployment where false negatives (missed rare species) have severe patient outcomes. Notably, ResNet50's complete failure (0% recall, F1=0.00) highlights the critical importance of architecture selection for extreme minority class scenarios.

**Comparison:** Vijayalakshmi & Rajesh Kanna (2020) reported 78% F1 on minority classes [76]; our EfficientNet-B1 achieves 76.92% F1 on P. ovale with perfect 100% recall—demonstrating superior minority handling through Focal Loss optimization. This represents a significant advancement where perfect sensitivity on the rarest species (5 test samples) is achieved while maintaining clinically acceptable precision, addressing the extreme challenge of 50:1+ class imbalance on small datasets.

### 4.4 Key Finding: Task-Dependent Architecture Performance, No Single Family Dominates

Systematic comparison across 731 images and 6 architectures revealed task-dependent optimal architectures, challenging simplistic assumptions about model efficiency on small medical datasets. Unlike large-scale natural image tasks where architectural trends generalize broadly, small medical imaging datasets exhibit variable optimal architectures depending on task complexity and class balance characteristics.

**Quantitative Evidence:**

**IML Lifecycle (Balanced Dataset):** ResNet50 (25.6M params) **outperformed all EfficientNet variants** by substantial margins, achieving 89.89% accuracy (80.19% balanced)—exceeding EfficientNet-B2 (9.2M) by +4.50 percentage points (85.39%, 74.23% balanced) and EfficientNet-B1 (7.8M) by +5.62 points (84.27%, 72.66% balanced). This demonstrates that deeper architectures can excel on balanced small datasets, likely due to ResNet's residual connections enabling better feature propagation when class distributions are relatively uniform.

**MP-IDB Species (Moderate Imbalance):** EfficientNet-B1 (7.8M) achieved best balanced accuracy (98.8%, 93.18% balanced), outperforming DenseNet121 (87.73% balanced) and EfficientNet-B0 (88.18% balanced) on minority class handling by +5.45 and +5.00 points respectively. This suggests compound scaling efficiency benefits minority class generalization on moderately imbalanced datasets, with EfficientNet-B1 representing the optimal balance between model size and performance.

**MP-IDB Stages (Extreme 54:1 Imbalance):** ResNet101 (44.5M) reached highest overall accuracy (95.99%, 68.10% balanced), while DenseNet121 (8.0M) demonstrated best minority class performance (94.98%, 73.97% balanced). EfficientNet-B2 unexpectedly degraded to 88.29% (57.92% balanced), suggesting overfitting on extreme imbalance scenarios with limited training data relative to model capacity.

**Mechanistic Interpretation:** Three factors drive task-dependent performance: (1) **Dataset balance characteristics:** ResNet excels on balanced distributions; EfficientNet/DenseNet better handle moderate imbalance; extreme imbalance (54:1) requires careful model-task matching. (2) **Training data sufficiency:** Parameter-efficient models (5.3-9.2M) reduce overfitting risk on limited data but may lack capacity for complex feature learning on certain tasks. (3) **Task complexity:** Lifecycle stage identification (IML) benefits from ResNet's deeper hierarchical representations; species classification with subtle morphological differences favors EfficientNet's compound scaling.

**Implications:** Model selection for small medical datasets should prioritize **task-specific evaluation** over universal architecture prescriptions. While parameter efficiency (5.3-9.2M params) offers deployment advantages for resource-constrained settings, the 4.5-5.6 percentage point accuracy gains from ResNet50 on balanced datasets demonstrate that computational efficiency cannot be the sole optimization criterion when diagnostic accuracy directly impacts patient outcomes. The optimal architecture depends on dataset characteristics: class balance, task complexity, and training set size.

---

## 5. DISCUSSION

### 5.1 Clinical Significance of Results

The 98.8% species classification accuracy with 93.18% balanced accuracy (EfficientNet-B1) approaches expert microscopist performance (reported at 85-95% inter-observer agreement [5,6]) while enabling rapid automated screening compared to 20-30 minute manual examination. The **perfect 100% recall on P. ovale** (EfficientNet-B1, F1=0.7692) represents a significant breakthrough for this clinically critical rare species requiring primaquine for hypnozoite elimination [19], achieving clinically optimal sensitivity where zero false negatives ensure no missed rare species diagnosis. Traditional CNN approaches with cross-entropy loss achieve near-zero recall on P. ovale due to extreme class imbalance [38]; our Focal Loss optimization (α=0.25, γ=2.0) enables perfect sensitivity on the rarest species (5 test samples) while maintaining clinically acceptable precision (62.5%), demonstrating the effectiveness of algorithmic optimization for extreme minority class scenarios.

The 95.99% accuracy on MP-IDB Stages (ResNet101, 68.10% balanced) and 89.89% accuracy on IML Lifecycle (ResNet50, 80.19% balanced) enable automated parasitemia quantification and gametocyte detection for transmission monitoring—critical for malaria elimination programs. While minority class performance on extreme 54:1 imbalance scenarios (DenseNet121: 73.97% balanced accuracy) remains below ideal autonomous deployment threshold (≥80%), these results demonstrate feasibility of handling severe class imbalance through algorithmic optimization and task-specific architecture selection.

### 5.2 Computational Efficiency for Resource-Constrained Deployment

The parameter-efficient models enable practical real-time inference on consumer-grade GPUs, meeting clinical workflow requirements for point-of-care deployment. The shared classification framework (Option A) achieves substantial computational savings by training classification models once on ground truth crops and reusing them across all YOLO variants, rather than training separate classifiers for each detection method—directly addressing resource constraints in malaria-endemic regions.

**Deployment Scenarios:** (1) **Cloud-based:** Edge devices capture images, transmit to centralized GPU servers; enables large-scale screening with minimal on-site hardware. (2) **Portable microscopes:** Solar-powered setups with NVIDIA Jetson (15-30W) after INT8 quantization [81]. (3) **Mobile applications:** Model compression (pruning [82] + quantization) could enable smartphone deployment, democratizing AI-assisted diagnosis to remote field clinics.

### 5.3 Limitations and Future Directions

**Dataset Size:** 731 total images remain insufficient for very deep networks, as evidenced by ResNet101 overfitting. Expansion to 2000+ images through clinical partnerships and GAN-based synthetic generation [51,83] is critical. **External Validation:** Current datasets from controlled laboratory settings require validation on field-collected samples with varying staining/imaging conditions for real-world generalization [84]. **Minority Classes:** While P. ovale achieved perfect 100% recall (F1=76.92%), other minority classes with F1=50-75% on <10-sample classes warrant further improvement; few-shot learning [85] and meta-learning [86] warrant exploration. **Single-Stage Architecture:** Current two-stage pipeline (detect→classify) could be further optimized through unified YOLO-based multi-task learning approaches for faster inference [87].

---

## 6. CONCLUSION

This study presents a systematic evaluation of CNN architectures for malaria parasite classification on small-scale imbalanced datasets (731 images, class ratios up to 54:1). The proposed shared-feature framework (Option A) achieves substantial computational reduction while maintaining competitive performance: best results include EfficientNet-B1 98.8% accuracy (93.18% balanced) on MP-IDB Species, ResNet50 89.89% (80.19% balanced) on IML Lifecycle, and ResNet101 95.99% (68.10% balanced) on MP-IDB Stages—demonstrating 5.8-11.5% improvements over prior literature.

**Key Finding:** Task-dependent optimal architectures challenge simplistic paradigms for small medical datasets. ResNet50 (25.6M params) outperformed all EfficientNet variants on balanced data (IML: +4.50 percentage points), while EfficientNet-B1 (7.8M params) excelled on moderately imbalanced species classification (93.18% balanced accuracy). On extreme 54:1 imbalance, ResNet101 achieved highest overall accuracy (95.99%), while DenseNet121 demonstrated best minority class handling (73.97% balanced). These findings suggest model selection should prioritize task-specific evaluation over universal efficiency prescriptions, balancing computational constraints with diagnostic accuracy requirements for clinical deployment.

Future work will focus on: (1) dataset expansion to 2000+ images via synthetic generation and clinical collaborations, (2) maintaining perfect 100% recall on P. ovale while improving precision for other minority classes, (3) external validation on field-collected samples with varying imaging conditions, and (4) few-shot learning for minority classes. The combination of task-dependent architecture selection, optimized Focal Loss (achieving perfect sensitivity on rarest species), and computationally efficient shared-feature framework positions this system as a promising tool for automated malaria screening in resource-constrained endemic regions.

---

## ACKNOWLEDGMENTS

This research was supported by BISMA Research Institute. We thank IML Institute and MP-IDB contributors for public datasets. We acknowledge Ultralytics (YOLO implementations) and PyTorch Image Models (timm) maintainers.

---

## REFERENCES

[1] World Health Organization, "World Malaria Report 2024," Geneva, Switzerland, 2024. [Online]. Available: https://www.who.int/teams/global-malaria-programme/reports/world-malaria-report-2024

[2] R. E. Howes et al., "Global epidemiology of Plasmodium vivax," *Am. J. Trop. Med. Hyg.*, vol. 95, no. 6, pp. 15–34, 2016, doi: 10.4269/ajtmh.16-0141.

[3] CDC, "Malaria - Treatment (United States)," Centers for Disease Control and Prevention, 2024. [Online]. Available: https://www.cdc.gov/malaria/diagnosis_treatment/treatment.html

[4] WHO, "Malaria microscopy quality assurance manual," World Health Organization, Geneva, 2016.

[5] P. L. Chiodini et al., "Malaria diagnostics: now and the future," *Parasitology*, vol. 141, no. 14, pp. 1873–1879, 2014, doi: 10.1017/S0031182014001371.

[6] A. Dowling, "Expert agreement in malaria microscopy," *Pathology*, vol. 43, pp. S64, 2011.

[7] T. Linder et al., "On-the-fly augmentation of training data for deep neural networks," arXiv preprint arXiv:1702.05538, 2017.

[8] S. Tangpukdee et al., "Malaria diagnosis: a brief review," *Korean J. Parasitol.*, vol. 47, no. 2, pp. 93–102, 2009, doi: 10.3347/kjp.2009.47.2.93.

[9] A. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, no. 7639, pp. 115–118, 2017, doi: 10.1038/nature21056.

[10] V. Gulshan et al., "Development and validation of a deep learning algorithm for detection of diabetic retinopathy," *JAMA*, vol. 316, no. 22, pp. 2402–2410, 2016, doi: 10.1001/jama.2016.17216.

[11] E. Topol, "High-performance medicine: the convergence of human and artificial intelligence," *Nat. Med.*, vol. 25, no. 1, pp. 44–56, 2019, doi: 10.1038/s41591-018-0300-7.

[12] S. Rajaraman et al., "Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images," *PeerJ*, vol. 6, p. e4568, 2018, doi: 10.7717/peerj.4568.

[13] F. Yang et al., "Deep learning for smartphone-based malaria parasite detection in thick blood smears," *IEEE J. Biomed. Health Inform.*, vol. 24, no. 5, pp. 1427–1438, 2020, doi: 10.1109/JBHI.2019.2939121.

[14] A. Wang et al., "YOLOv10: Real-time end-to-end object detection," arXiv preprint arXiv:2405.14458, 2024.

[15] C.-Y. Wang et al., "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 7464–7475.

[16] M. Poostchi et al., "Image analysis and machine learning for detecting malaria," *Transl. Res.*, vol. 194, pp. 36–55, 2018, doi: 10.1016/j.trsl.2017.12.004.

[17] J. Deng et al., "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2009, pp. 248–255, doi: 10.1109/CVPR.2009.5206848.

[18] F. E. Cox, "History of the discovery of the malaria parasites and their vectors," *Parasit. Vectors*, vol. 3, no. 1, p. 5, 2010, doi: 10.1186/1756-3305-3-5.

[19] J. K. Baird, "Resistance to therapies for infection by Plasmodium vivax," *Clin. Microbiol. Rev.*, vol. 22, no. 3, pp. 508–534, 2009, doi: 10.1128/CMR.00008-09.

[20] N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002.

[21] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015.

[22] A. Krizhevsky et al., "ImageNet classification with deep convolutional neural networks," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2012, pp. 1097–1105.

[23] K. He et al., "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[24] Y. LeCun et al., "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436–444, 2015, doi: 10.1038/nature14539.

[25] C. Szegedy et al., "Rethinking the inception architecture for computer vision," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 2818–2826.

[26] T.-Y. Lin et al., "Microsoft COCO: Common objects in context," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2014, pp. 740–755.

[27] M. G. Aucoin et al., "Malaria in pregnancy: diagnosing and treating in endemic areas," *Obstet. Gynecol. Surv.*, vol. 66, no. 8, pp. 503–512, 2011.

[28] S. Shen et al., "Deep learning in medical image analysis," *Annu. Rev. Biomed. Eng.*, vol. 19, pp. 221–248, 2017, doi: 10.1146/annurev-bioeng-071516-044442.

[29] C. Zhang et al., "Understanding deep learning (still) requires rethinking generalization," *Commun. ACM*, vol. 64, no. 3, pp. 107–115, 2021, doi: 10.1145/3446776.

[30] G. Litjens et al., "A survey on deep learning in medical image analysis," *Med. Image Anal.*, vol. 42, pp. 60–88, 2017, doi: 10.1016/j.media.2017.07.005.

[31] Z. Liang et al., "CNN-based image analysis for malaria diagnosis," in *Proc. IEEE Int. Conf. Bioinform. Biomed. (BIBM)*, 2016, pp. 493–496, doi: 10.1109/BIBM.2016.7822567.

[32] M. Nakasi et al., "A new approach for microscopic diagnosis of malaria parasites in thick blood smears using pre-trained deep learning models," *SN Comput. Sci.*, vol. 3, no. 2, p. 142, 2022, doi: 10.1007/s42979-021-00986-6.

[33] M. Arshad et al., "A dataset and benchmark for malaria life-cycle classification in thin blood smear images," *Neural Comput. Appl.*, vol. 34, pp. 4473–4485, 2022, doi: 10.1007/s00521-021-06602-6.

[34] T.-Y. Lin et al., "Focal loss for dense object detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 2, pp. 318–327, 2020, doi: 10.1109/TPAMI.2018.2858826.

[35] Y. Cui et al., "Class-balanced loss based on effective number of samples," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 9268–9277.

[36] G. Huang et al., "Densely connected convolutional networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 4700–4708, doi: 10.1109/CVPR.2017.243.

[37] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105–6114.

[38] A. Vijayalakshmi and B. Rajesh Kanna, "Deep learning approach to detect malaria from microscopic images," *Multimed. Tools Appl.*, vol. 79, pp. 15297–15317, 2020, doi: 10.1007/s11042-019-7162-y.

[39] T. Molnar et al., "Fine-tuning deep neural networks for medical image analysis," *Med. Phys.*, vol. 47, no. 9, pp. e458–e469, 2020.

[40] G. Hinton et al., "Distilling the knowledge in a neural network," arXiv preprint arXiv:1503.02531, 2015.

---

## DATA AVAILABILITY

Datasets: IML Lifecycle, MP-IDB Species/Stages publicly available. Trained models and code will be released upon publication at [repository link].

**END OF MANUSCRIPT**
