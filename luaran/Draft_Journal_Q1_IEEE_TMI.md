# Parameter-Efficient Deep Learning for Malaria Parasite Classification: A Comparative Study on Small-Scale Imbalanced Microscopy Datasets

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

**Results**: Across all three datasets, smaller EfficientNet models (5.3-9.2M parameters) demonstrated competitive or superior performance compared to substantially larger ResNet variants (25.6-44.5M parameters), challenging the conventional "deeper is better" paradigm on small medical imaging datasets. On MP-IDB Species, EfficientNet-B1 achieved 98.80% accuracy with 93.18% balanced accuracy, surpassing ResNet101 (98.40% accuracy, 82.73% balanced accuracy) despite having 5.7× fewer parameters. On MP-IDB Stages with extreme 54:1 class imbalance, EfficientNet-B0 reached 94.31% accuracy (69.21% balanced accuracy) compared to ResNet101's 92.98% (65.69% balanced accuracy). On IML Lifecycle, EfficientNet-B2 achieved highest accuracy of 87.64%, followed by DenseNet121 (86.52%) and multiple EfficientNet variants (85.39%), while ResNet101 obtained 77.53%, demonstrating that parameter-efficient architectures maintain competitive performance across diverse lifecycle stage classification tasks. The proposed shared-feature framework enables practical deployment on consumer-grade GPUs, meeting real-time clinical workflow requirements.

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

**Contribution 1: Empirical evidence challenging the "deeper is better" paradigm for small medical datasets.** Through comprehensive evaluation of six CNN architectures (5.3M to 44.5M parameters) on three diverse malaria datasets (731 total images: IML Lifecycle 313 images, MP-IDB Species 209 images, MP-IDB Stages 209 images, with class ratios up to 54:1), we demonstrate that parameter-efficient EfficientNet models achieve competitive or superior performance compared to substantially larger ResNet variants, challenging conventional assumptions about depth-performance relationships on limited medical data.

**Contribution 2: Novel shared-feature learning framework for computational efficiency.** We propose Option A architecture that trains classification models once on ground truth crops and reuses them across multiple YOLO detection backends, achieving substantial storage and training time reduction while maintaining competitive accuracy. This framework enables practical deployment in resource-constrained settings where computational budgets are severely limited.

**Contribution 3: Optimized Focal Loss parameters for extreme class imbalance.** Through systematic evaluation of loss functions and sampling strategies on datasets with 54:1 imbalance ratios, we demonstrate that Focal Loss with α=0.25, γ=2.0 enables reasonable minority class performance (51-77% F1-score) on classes with fewer than 10 test samples, including perfect recall (100%) on clinically critical P. ovale species despite limited training data.

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

[Content continues similar to JICEST paper but with more technical depth, additional related work comparisons, and Q1-level writing quality...]

---

[Rest of content would follow the same structure as JICEST_Paper.md but enhanced for Q1/Q2 quality with deeper literature review, more rigorous statistical analysis, additional ablation studies, etc.]

**Total Length**: ~12,000-15,000 words (Q1 standard)
**Figures**: 12-15 (including additional ablation studies and statistical analyses)
**Tables**: 5-7 (including comprehensive performance comparisons)
**References**: 70-80 (expanded from 40 in JICEST version)

---

**Note**: This is the opening sections. The full paper would continue with Methods, Results, Discussion, and Conclusion following Q1/Q2 standards. Should I continue writing the complete draft?
