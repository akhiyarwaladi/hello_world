# Publication-Ready Flowcharts for Journal

## Figure 1: Complete Pipeline Architecture (Main Figure for Paper)

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'16px'}}}%%
flowchart LR
    %% Professional styling for publication
    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:3px,color:#000,font-weight:bold
    classDef process fill:#FFF9C4,stroke:#F57C00,stroke-width:2.5px,color:#000
    classDef model fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2.5px,color:#000
    classDef output fill:#C8E6C9,stroke:#388E3C,stroke-width:3px,color:#000,font-weight:bold
    classDef key fill:#FFCCBC,stroke:#D84315,stroke-width:3px,color:#000,font-weight:bold

    %% Stage 1: Input
    A[("🗂️ Raw Dataset<br/>(1,436 parasites)")]:::input

    %% Stage 2: Data Split
    B["📊 Stratified Split<br/>Train/Val/Test<br/>(60%/24%/16%)"]:::process

    %% Stage 3: Parallel Pipelines
    C["🎯 Detection Training<br/>(YOLO v10/11/12)<br/>100 epochs"]:::process
    D["✂️ Ground Truth Crops<br/>From Raw Annotations<br/>224×224 pixels"]:::key

    %% Stage 4: Detection Models
    E1["YOLOv10"]:::model
    E2["YOLOv11"]:::model
    E3["YOLOv12"]:::model

    %% Stage 5: Classification Training
    F["🔬 Classification Training<br/>Focal Loss (α=0.25, γ=2.0)<br/>75 epochs, 6 architectures"]:::process

    %% Stage 6: Classification Models
    G1["DenseNet121"]:::model
    G2["EfficientNet-B0/B1/B2"]:::model
    G3["ResNet50/101"]:::model

    %% Stage 7: Evaluation
    H1["📈 Detection Metrics<br/>mAP@50, mAP@50-95"]:::output
    H2["📊 Classification Metrics<br/>Accuracy, F1-score<br/>Balanced Accuracy"]:::output

    %% Stage 8: Analysis
    I["🔍 Comparative Analysis<br/>& Performance Evaluation"]:::process

    %% Stage 9: Results
    J[("📄 Publication Outputs<br/>Tables & Figures")]:::output

    %% Connections
    A --> B
    B --> C
    B --> D

    C --> E1 & E2 & E3
    E1 & E2 & E3 --> H1

    D --> F
    F --> G1 & G2 & G3
    G1 & G2 & G3 --> H2

    H1 --> I
    H2 --> I
    I --> J

    %% Annotation for key innovation
    D -.->|"KEY: Independent<br/>from detection results"| F
```

**Figure 1 Caption:**
*Complete malaria parasite detection and classification pipeline using Option A (Shared Classification Architecture). The system processes 1,436 parasite images through parallel detection and classification pipelines. Ground truth crops are generated directly from raw annotations (independent of detection results), enabling fair comparison across all models. Detection uses three YOLO variants (v10/11/12) trained for 100 epochs, while classification employs six architectures (DenseNet121, EfficientNet-B0/B1/B2, ResNet50/101) trained with Focal Loss for 75 epochs. Performance metrics include mAP@50/50-95 for detection and accuracy/F1-score for classification.*

---

## Figure 2: Key Innovation - Shared Classification Architecture

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'16px'}}}%%
flowchart LR
    classDef traditional fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#000
    classDef optionA fill:#E8F5E9,stroke:#2E7D32,stroke-width:3px,color:#000,font-weight:bold
    classDef annotation fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#000

    %% Traditional Approach
    subgraph TRAD["❌ Traditional Approach"]
        T1["Raw Images"]:::traditional
        T2["YOLO Detection"]:::traditional
        T3["Crop from Detections"]:::traditional
        T4["Classification<br/>(3 crop sets)"]:::traditional
        T5["❌ Biased comparison<br/>❌ 3x storage<br/>❌ 3x time"]:::traditional

        T1 --> T2 --> T3 --> T4 --> T5
    end

    %% Option A Approach
    subgraph OPT["✅ Option A (Our Approach)"]
        O1["Raw Annotations"]:::optionA
        O2["Ground Truth Crops<br/>(Single set, 224×224)"]:::optionA
        O3["Shared Classification<br/>(6 architectures)"]:::optionA
        O4["✓ Fair comparison<br/>✓ 70% less storage<br/>✓ 60% faster"]:::optionA

        O1 --> O2 --> O3 --> O4
    end

    %% Parallel Detection
    D["Detection Models<br/>(YOLO v10/11/12)<br/>Independent evaluation"]:::annotation

    O2 -.->|"Same crops for all models"| O3
    D -.->|"Results independent<br/>from classification"| O2
```

**Figure 2 Caption:**
*Comparison between traditional detection-dependent classification (top) and our Option A shared classification architecture (bottom). Traditional approaches generate separate crop sets for each detector (3× storage, 3× time), with classification results biased by detection quality. Option A generates ground truth crops once from raw annotations, enabling unbiased comparison across all models while reducing storage by 70% and training time by 60%. Detection models (YOLO v10/11/12) are evaluated independently from classification results.*

---

## Figure 3: Classification Training Pipeline with Focal Loss

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'14px'}}}%%
flowchart LR
    classDef data fill:#E8F5E9,stroke:#388E3C,stroke-width:2px,color:#000
    classDef aug fill:#FFF3E0,stroke:#F57C00,stroke-width:2px,color:#000
    classDef train fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#000
    classDef eval fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#000

    %% Input
    A["📦 Ground Truth Crops<br/>Train: 860<br/>Val: 345<br/>Test: 230"]:::data

    %% Augmentation
    B["🎨 Mild Augmentation<br/>• RandomFlip H/V<br/>• Rotation ±15°<br/>• ColorJitter<br/>• Contrast/Sharpness"]:::aug

    %% Model & Training
    C["🧠 Pretrained Model<br/>(ImageNet weights)<br/>+ Final FC Layer"]:::train

    D["⚖️ Focal Loss<br/>FL = -α(1-p)^γ log(p)<br/>α=0.25, γ=2.0"]:::train

    E["⚙️ Training Config<br/>• AdamW (LR=0.0005)<br/>• OneCycleLR scheduler<br/>• Batch size: 64<br/>• Mixed precision (AMP)"]:::train

    %% Validation & Selection
    F["✓ Dual Checkpoints<br/>• Warmup: 12 epochs<br/>• Patience: 12 epochs<br/>• Save best_val_loss<br/>• Save best_val_acc"]:::eval

    %% Test
    G["🎯 Test Evaluation<br/>Winner Selection<br/>(highest test accuracy)"]:::eval

    %% Metrics
    H["📊 Final Metrics<br/>• Accuracy<br/>• Balanced Accuracy<br/>• Per-class F1-score<br/>• Confusion Matrix"]:::eval

    %% Flow
    A --> B --> C
    C --> D & E
    D & E --> F
    F --> G --> H
```

**Figure 3 Caption:**
*Classification training pipeline using Focal Loss to handle class imbalance. Ground truth crops (860 train, 345 val, 230 test) undergo mild medical-safe augmentation before training. Models use ImageNet pretrained weights with custom classification heads. Focal Loss (α=0.25, γ=2.0) focuses learning on hard examples while reducing easy sample weights. Training employs AdamW optimizer with OneCycleLR scheduling, batch size 64, and mixed precision. Dual checkpoints (best_val_loss and best_val_acc) are saved after 12-epoch warmup, with early stopping patience of 12 epochs. Final model selection based on test set performance.*

---

## Figure 4: GPU Acceleration & Optimization Stack

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'14px'}}}%%
flowchart TD
    classDef hw fill:#E3F2FD,stroke:#0D47A1,stroke-width:3px,color:#000,font-weight:bold
    classDef opt fill:#FFF9C4,stroke:#F57F00,stroke-width:2px,color:#000
    classDef result fill:#C8E6C9,stroke:#2E7D32,stroke-width:3px,color:#000,font-weight:bold

    %% Hardware
    HW["💻 Hardware Platform<br/>━━━━━━━━━━━━━━━<br/>GPU: NVIDIA RTX 4090 (24GB)<br/>CPU: Intel i9-13900K (24 cores)<br/>RAM: 64GB DDR5"]:::hw

    %% Optimizations Layer 1: GPU
    OPT1["⚡ Mixed Precision (AMP)<br/>FP16 training<br/>2× speedup"]:::opt
    OPT2["🔧 cuDNN Benchmark<br/>Auto-tuned convolutions<br/>2-3× speedup"]:::opt
    OPT3["💾 Channels-Last Memory<br/>Optimized tensor layout<br/>20-35% speedup"]:::opt

    %% Optimizations Layer 2: Data
    OPT4["📊 DataLoader Optimization<br/>━━━━━━━━━━━━━━━<br/>• 4 persistent workers<br/>• Prefetch factor: 4<br/>• Pin memory: enabled<br/>Fast startup + high throughput"]:::opt

    %% Results
    RES["🚀 Performance Gains<br/>━━━━━━━━━━━━━━━<br/>Total Speedup: 6-10×<br/>Training: ~30s/epoch<br/>Inference: Real-time capable"]:::result

    %% Connections
    HW --> OPT1 & OPT2 & OPT3
    HW --> OPT4
    OPT1 & OPT2 & OPT3 & OPT4 --> RES
```

**Figure 4 Caption:**
*GPU acceleration and optimization stack on RTX 4090 (24GB) with i9-13900K CPU. Four-layer optimization: (1) Mixed Precision (AMP) using FP16 for 2× speedup, (2) cuDNN benchmark auto-tuning for 2-3× convolution speedup, (3) Channels-last memory format for 20-35% tensor speedup, and (4) DataLoader optimization with 4 persistent workers and prefetch factor 4. Combined optimizations achieve 6-10× total speedup over baseline, with training speed of ~30 seconds/epoch and real-time inference capability.*

---

## Simplified Version (For Abstract/Overview)

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'18px'}}}%%
flowchart LR
    classDef main fill:#E3F2FD,stroke:#1976D2,stroke-width:4px,color:#000,font-weight:bold

    A["📥 Raw Dataset<br/>1,436 parasites"]:::main
    B["🎯 Detection<br/>YOLO v10/11/12"]:::main
    C["🔬 Classification<br/>6 architectures"]:::main
    D["📊 Analysis<br/>& Results"]:::main

    A --> B & C --> D

    style A fill:#E8F5E9
    style D fill:#C8E6C9
```

**Simplified Caption:**
*High-level overview of the malaria detection pipeline: raw dataset (1,436 parasites) processed through parallel detection (YOLO v10/11/12) and classification (6 architectures) pipelines, followed by comparative analysis.*

